#include "cuquanpath.h"

void QuanPath(QCircuit &qc)
{
    CudaUtils cudaUtils;
    // 将量子线路存入GPU内存
    qc.copyQCircuitToDevice();

    // 在const memory初始化门矩阵
    HANDLE_CUDA_ERROR(initGateMatricesInConstMemory());

    // 分配初始状态向量|00..0>的内存空间
    DTYPE *deviceSv;
    ll stateVectorLen = 1 << qc.numQubits;
    size_t stateVectorBytes = stateVectorLen * sizeof(DTYPE);
    HANDLE_CUDA_ERROR(cudaMalloc(&deviceSv, stateVectorBytes));

    // 分配每一层的高阶矩阵的内存空间
    DTYPE *highMatrices;
    ll highMatrixElementNum = (1 << qc.numHighQubits) * (1 << qc.numHighQubits);
    size_t highMatrixBytes = highMatrixElementNum * qc.numDepths * sizeof(DTYPE);
    HANDLE_CUDA_ERROR(cudaMalloc(&highMatrices, highMatrixBytes));

    // 高阶计算每一层的张量积
    dim3 blockDim(1 << qc.numHighQubits, 1 << qc.numHighQubits);
    // cudaUtils.startTiming("highTensorProduct");
    highTensorProduct<<<qc.numDepths, blockDim>>>(qc.d_gate_array, highMatrices, qc.numDepths, qc.numQubits, qc.numHighQubits);

    // 矩阵乘法
    DTYPE *highFinalMatrix;
    size_t highFinalMatrixBytes = highMatrixElementNum * sizeof(DTYPE);
    cudaMalloc(&highFinalMatrix, highFinalMatrixBytes);
    cudaMemcpy(highFinalMatrix, highMatrices, highFinalMatrixBytes, cudaMemcpyDeviceToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);
    const cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
    const cuDoubleComplex beta = make_cuDoubleComplex(0.0, 0.0);

    DTYPE *tempMat;
    cudaMalloc(&tempMat, highMatrixElementNum * sizeof(cuDoubleComplex));

    int N = 1 << qc.numHighQubits;
    for (int i = 1; i < qc.numDepths; i++)
    {
        // A = highFinalMatrix, B = highMatrices[i], C = tempMat
        cublasZgemm(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            N, N, N,
            &alpha,
            highFinalMatrix, N,
            highMatrices + i * N * N, N,
            &beta,
            tempMat, N);
        // Swap pointers
        DTYPE *tmp = highFinalMatrix;
        highFinalMatrix = tempMat;
        tempMat = tmp;
    }
    // 释放 cuBLAS 句柄
    cublasDestroy(handle);
    cudaDeviceSynchronize();

    // cudaUtils.stopTiming();

    // 检查高阶结果
    // cuDoubleComplex *hostResult = (cuDoubleComplex *)malloc(highMatrixElementNum * sizeof(cuDoubleComplex));
    // cudaMemcpy(hostResult, highFinalMatrix, highMatrixElementNum * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
    // printf("Final high matrix result:\n");
    // for (int row = 0; row < N; row++)
    // {
    //     for (int col = 0; col < N; col++)
    //     {
    //         cuDoubleComplex val = hostResult[row * N + col];
    //         printf("(% .3f%+.3fi) ", cuCreal(val), cuCimag(val));
    //     }
    //     printf("\n");
    // }

    // Step 2. Local SVSim for gates on low-order qubits
    int threadPerBlock = 1 << (qc.numLowQubits - 1);
    assert(threadPerBlock <= MAX_THREAD_PER_BLOCK);
    int blockPerGrid = (stateVectorLen + threadPerBlock - 1) / (threadPerBlock * 2);
    size_t sharedMemSize = (stateVectorLen / blockPerGrid) * sizeof(DTYPE);
    assert(sharedMemSize <= MAX_SHARED_MEMORY);

    // cudaUtils.startTiming("SVSim");
    SVSim<<<blockPerGrid, threadPerBlock, sharedMemSize>>>(qc.d_gate_array, deviceSv, stateVectorLen / blockPerGrid, qc.numDepths, qc.numQubits, qc.numLowQubits);
    cudaDeviceSynchronize();
    // cudaUtils.stopTiming();

    
    // Step 3. Final merge that requires communication
    int localSvLen = stateVectorLen / (1 << qc.numHighQubits);
    int numMergeThreads = blockPerGrid * 128;
    int numElementsPerThread = (stateVectorLen + numMergeThreads - 1) / numMergeThreads;
    // cudaUtils.startTiming("merge");
    merge<<<blockPerGrid, 128>>>(deviceSv, highFinalMatrix, stateVectorLen, 1 << qc.numHighQubits, localSvLen, numElementsPerThread);
    cudaDeviceSynchronize();

    // cudaUtils.stopTiming();
    // cudaUtils.writeDeviceStateVectorToFile(deviceSv, stateVectorLen);

    // 释放设备内存
    HANDLE_CUDA_ERROR(cudaFree(deviceSv));
    HANDLE_CUDA_ERROR(cudaFree(highMatrices));
    HANDLE_CUDA_ERROR(cudaFree(highFinalMatrix));
    HANDLE_CUDA_ERROR(cudaFree(tempMat));
}

__global__ void highTensorProduct(QGateDevice *d_gate_array, DTYPE *ptrOpmat, int numDepths, int numQubits, int numHighQubits)
{
    int depthIdx = blockIdx.x;
    int row = threadIdx.y;
    int col = threadIdx.x;
    int matrixSize = 1 << numHighQubits;
    // if (depthIdx == 0)
    //     printf("depthIdx=%d, row=%d, col=%d \n", depthIdx, row, col);

    __shared__ DTYPE sharedGates[4][2 * 2];
    if (row == 0 && col == 0)
    {
        for (int qid = numQubits - 1; qid >= numQubits - numHighQubits; qid--)
        {
            int gateIdx = depthIdx * numQubits + qid;
            QGateDevice gate = d_gate_array[gateIdx];
            // printf("gate[%d].thera = %f",gateIdx, gate.theta );
            const DTYPE *gateMat = get_matrix(gate);
            for (int i = 0; i < 4; i++)
            {
                sharedGates[numQubits - qid - 1][i] = gateMat[i];
                // printf("sharedGates[%d][%d] = (%f, %f)\n", numQubits - qid - 1, i, cuCreal(sharedGates[numQubits - qid - 1][i]), cuCimag(sharedGates[numQubits - qid - 1][i]));
            }
        }
    }
    __syncthreads();

    DTYPE temp = make_cuDoubleComplex(1, 0);
    for (int i = numHighQubits - 1; i >= 0; i--)
    {
        int bitR = (row >> i) & 1;
        int bitC = (col >> i) & 1;
        temp = myCmul(sharedGates[numHighQubits - 1 - i][bitR * 2 + bitC], temp);
    }
    ptrOpmat[depthIdx * (matrixSize * matrixSize) + row * matrixSize + col] = temp;
    // __syncthreads();
    // if (depthIdx == 0)
    // {
    //     if (threadIdx.y < matrixSize && threadIdx.x < matrixSize)
    //     {
    //         int r = threadIdx.y;
    //         int c = threadIdx.x;
    //         DTYPE val = ptrOpmat[depthIdx * (matrixSize * matrixSize) + r * matrixSize + c];
    //         printf("Opmat[%d][%d] = (%f, %f)\n", r, c, cuCreal(val), cuCimag(val));
    //     }
    // }

    return;
}

__global__ void SVSim(QGateDevice *d_gate_array, DTYPE *deviceSv, int numStatePerBlock, int numDepths, int numQubits, int numLowQubits)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ DTYPE shared_state[];
    for (int i = threadIdx.x; i < numStatePerBlock; i += blockDim.x)
    {
        shared_state[i] = deviceSv[blockIdx.x * numStatePerBlock + i];
    }

    if (idx == 0)
        shared_state[0] = make_cuDoubleComplex(1, 0);
    __syncthreads();

    for (int lev = 0; lev < numDepths; ++lev)
    {
        for (int qid = 0; qid < numLowQubits; ++qid)
        {
            int gate_idx = lev * numQubits + qid;
            QGateDevice gate = d_gate_array[gate_idx];

            if (gate.gname_id == 1)
                continue;

            DTYPE state[2];

            // single qubit
            if (gate.numTargetQubits == 1)
            {
                int low_bit_mask = gate.strides[1] - 1;
                int high_bits = threadIdx.x & ~low_bit_mask;
                int low_bits = threadIdx.x & low_bit_mask;

                // 计算这一对 index
                int i0 = (high_bits << 1) | low_bits;
                int i1 = i0 | gate.strides[1];

                state[0] = shared_state[i0];
                state[1] = shared_state[i1];

                int globalIdx0 = blockIdx.x * numStatePerBlock + i0;
                int globalIdx1 = blockIdx.x * numStatePerBlock + i1;
                if (isLegalControlPattern(globalIdx0, gate) && isLegalControlPattern(globalIdx1, gate))
                {
                    const DTYPE *gateMat = get_matrix(gate);
                    shared_state[i0] = myCadd(myCmul(gateMat[0], state[0]), myCmul(gateMat[1], state[1]));
                    shared_state[i1] = myCadd(myCmul(gateMat[2], state[0]), myCmul(gateMat[3], state[1]));
                }
            }
            __syncthreads();
        }
    }

    // 将共享内存写回全局内存./
    for (int i = threadIdx.x; i < numStatePerBlock; i += blockDim.x)
    {
        deviceSv[blockIdx.x * numStatePerBlock + i] = shared_state[i];
    }
}

__device__ bool isLegalControlPattern(int qid, QGateDevice &gate)
{

    for (int i = 0; i < gate.numControlQubits; ++i)
    {
        // [TODO] Check the control qubits of the gate ////////////////
        // [HINT] If the i-th bit of amp is 0 and q_i is a |1> control qubit of gate, return false.
        // cout << "[TODO] Check the control qubits of the gate." << endl;
        // exit(1);

        // 1-controlled and the control qubit of amp is 0
        if ((qid & gate.ctrlmask) == 0)
        {
            return false;
        }
        // ///////////////////////////////////////////////////////////
    }
    return true;
}

__global__ void SVSimForSingleQubit(Matrix<DTYPE> *gateMatrix, int numLowQubits, Matrix<DTYPE> *localSv, int qidx)
{
    // 将 gateMatrix 加载到共享内存
    __shared__ DTYPE sharedGateMatrix[2][2];
    if (threadIdx.x < 4)
    {
        int row = threadIdx.x / 2;
        int col = threadIdx.x % 2;
        sharedGateMatrix[row][col] = gateMatrix->data[row][col];
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < localSv->row / 2)
    {
        int i = (idx / (1 << qidx)) * (1 << (qidx + 1));
        int j = idx % (1 << qidx);
        int p = i | j;
        DTYPE q0 = localSv->data[p][0];
        DTYPE q1 = localSv->data[p | 1 << qidx][0];
        localSv->data[p][0] = cuCadd(cuCmul(sharedGateMatrix[0][0], q0), cuCmul(sharedGateMatrix[0][1], q1));
        localSv->data[p | 1 << qidx][0] = cuCadd(cuCmul(sharedGateMatrix[1][0], q0), cuCmul(sharedGateMatrix[1][1], q1));
    }
}

/**
 * @brief Conduct SVSim for gate on two qubits
 *
 * @param localSv the local state vector pointer
 * @param gateMatrix the gate matrix pointer
 * @param numLowQubits the number of low-order qubits
 * @param qlow low index of target qubit
 * @param qhigh high index of target qubit
 */
__global__ void SVSimForTwoQubit(Matrix<DTYPE> *gateMatrix, int numLowQubits, Matrix<DTYPE> *localSv, int qlow, int qhigh)
{
    // 将 gateMatrix 加载到共享内存
    __shared__ DTYPE sharedGateMatrix[4][4];
    if (threadIdx.x < 16)
    {
        int row = threadIdx.x / 4;
        int col = threadIdx.x % 4;
        sharedGateMatrix[row][col] = gateMatrix->data[row][col];
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < localSv->row / 4)
    {
        int i = (idx / (1 << (qhigh - 1))) * (1 << (qhigh + 1));
        int j = (((idx / (1 << (qhigh - 1))) << (qhigh - qlow - 1)) ^ (idx / (1 << qlow))) * (1 << (qlow + 1));
        int k = idx % (1 << qlow);
        int p = i | j | k;

        DTYPE q0 = localSv->data[p][0];
        DTYPE q1 = localSv->data[p | 1 << qlow][0];
        DTYPE q2 = localSv->data[p | 1 << qhigh][0];
        DTYPE q3 = localSv->data[p | 1 << qlow | 1 << qhigh][0];

        localSv->data[p][0] = cuCadd(cuCadd(cuCmul(sharedGateMatrix[0][0], q0), cuCmul(sharedGateMatrix[0][1], q1)), cuCadd(cuCmul(sharedGateMatrix[0][2], q2), cuCmul(sharedGateMatrix[0][3], q3)));
        localSv->data[p | (1 << qlow)][0] = cuCadd(cuCadd(cuCmul(sharedGateMatrix[1][0], q0), cuCmul(sharedGateMatrix[1][1], q1)), cuCadd(cuCmul(sharedGateMatrix[1][2], q2), cuCmul(sharedGateMatrix[1][3], q3)));
        localSv->data[p | (1 << qhigh)][0] = cuCadd(cuCadd(cuCmul(sharedGateMatrix[2][0], q0), cuCmul(sharedGateMatrix[2][1], q1)), cuCadd(cuCmul(sharedGateMatrix[2][2], q2), cuCmul(sharedGateMatrix[2][3], q3)));
        localSv->data[p | (1 << qlow) | (1 << qhigh)][0] = cuCadd(cuCadd(cuCmul(sharedGateMatrix[3][0], q0), cuCmul(sharedGateMatrix[3][1], q1)), cuCadd(cuCmul(sharedGateMatrix[3][2], q2), cuCmul(sharedGateMatrix[3][3], q3)));
    }
}

/**
 * @brief [TODO] Conduct the final merge operation in QuanPath
 *
 * @param sv the state vector
 * @param ptrOpmat the pointer to the high-order operation matrix
 */
__global__ void merge(DTYPE *deviceSv, DTYPE *ptrOpmat, int svLen, int highMatrixSize, int localSvLen, int numElementsPerThread)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // int localSvLen = sv->row / opmatSize;
    // int totalThreadsNum = blockDim.x * gridDim.x;
    // Calculate the number of elements each thread should process
    // int numElementsPerThread = (sv->row + totalThreadsNum - 1) / totalThreadsNum;

    // Calculate the starting index for the current thread
    int startIdx = idx * numElementsPerThread;

    // printf("idx = % d , startIdx = %d;\n",idx, startIdx);

    // Loop through the elements this thread is responsible for
    for (int k = 0; k < numElementsPerThread; ++k)
    {
        int currentIdx = startIdx + k;

        if (currentIdx < svLen)
        {
            DTYPE ans = make_cuDoubleComplex(0, 0);
            for (ll i = 0; i < highMatrixSize; i++)
            {
                ans = cuCadd(ans, cuCmul(ptrOpmat[(currentIdx / localSvLen) * highMatrixSize + i], deviceSv[currentIdx % localSvLen + localSvLen * i]));
            }
            deviceSv[currentIdx] = ans;
            // printf("deviceSv[%d] = %f + %f i;",currentIdx, ans.x, ans.y );
        }
        __syncthreads();
    }
}

__device__ __forceinline__ cuDoubleComplex myCmul(cuDoubleComplex a, cuDoubleComplex b)
{
    return make_cuDoubleComplex(
        a.x * b.x - a.y * b.y,
        a.x * b.y + a.y * b.x);
}

__device__ __forceinline__ cuDoubleComplex myCadd(cuDoubleComplex a, cuDoubleComplex b)
{
    return make_cuDoubleComplex(
        a.x + b.x,
        a.y + b.y);
}

// 初始化门矩阵到常量内存的函数
cudaError_t initGateMatricesInConstMemory()
{

    // IDE矩阵 (单位矩阵)
    DTYPE h_IDE[4] = {
        make_cuDoubleComplex(1.0, 0.0), make_cuDoubleComplex(0.0, 0.0),
        make_cuDoubleComplex(0.0, 0.0), make_cuDoubleComplex(1.0, 0.0)};

    // H矩阵 (Hadamard门)
    double invSqrt2 = 1.0 / sqrt(2.0);
    DTYPE h_H[4] = {
        make_cuDoubleComplex(invSqrt2, 0.0), make_cuDoubleComplex(invSqrt2, 0.0),
        make_cuDoubleComplex(invSqrt2, 0.0), make_cuDoubleComplex(-invSqrt2, 0.0)};

    // X矩阵 (Pauli-X门)
    DTYPE h_X[4] = {
        make_cuDoubleComplex(0.0, 0.0), make_cuDoubleComplex(1.0, 0.0),
        make_cuDoubleComplex(1.0, 0.0), make_cuDoubleComplex(0.0, 0.0)};

    // Y矩阵 (Pauli-Y门)
    DTYPE h_Y[4] = {
        make_cuDoubleComplex(0.0, 0.0), make_cuDoubleComplex(0.0, -1.0),
        make_cuDoubleComplex(0.0, 1.0), make_cuDoubleComplex(0.0, 0.0)};

    // Z矩阵 (Pauli-Z门)
    DTYPE h_Z[4] = {
        make_cuDoubleComplex(1.0, 0.0), make_cuDoubleComplex(0.0, 0.0),
        make_cuDoubleComplex(0.0, 0.0), make_cuDoubleComplex(-1.0, 0.0)};

    DTYPE h_SWAP[16] = {
        make_cuDoubleComplex(1.0, 0.0), make_cuDoubleComplex(0.0, 0.0), make_cuDoubleComplex(0.0, 0.0), make_cuDoubleComplex(0.0, 0.0),
        make_cuDoubleComplex(0.0, 0.0), make_cuDoubleComplex(0.0, 0.0), make_cuDoubleComplex(1.0, 0.0), make_cuDoubleComplex(0.0, 0.0),
        make_cuDoubleComplex(0.0, 0.0), make_cuDoubleComplex(1.0, 0.0), make_cuDoubleComplex(0.0, 0.0), make_cuDoubleComplex(0.0, 0.0),
        make_cuDoubleComplex(0.0, 0.0), make_cuDoubleComplex(0.0, 0.0), make_cuDoubleComplex(0.0, 0.0), make_cuDoubleComplex(1.0, 0.0)};

    // 将矩阵数据复制到常量内存
    cudaMemcpyToSymbol(d_IDE, h_IDE, sizeof(DTYPE) * 4);
    cudaMemcpyToSymbol(d_H, h_H, sizeof(DTYPE) * 4);
    cudaMemcpyToSymbol(d_X, h_X, sizeof(DTYPE) * 4);
    cudaMemcpyToSymbol(d_Y, h_Y, sizeof(DTYPE) * 4);
    cudaMemcpyToSymbol(d_Z, h_Z, sizeof(DTYPE) * 4);
    cudaMemcpyToSymbol(d_SWAP, h_SWAP, sizeof(DTYPE) * 16);

    return cudaSuccess;
}

__device__ __forceinline__ const DTYPE *get_matrix(QGateDevice &gate)
{
    if (gate.theta != 0)
    {
        // 计算 sin 和 cos 值
        double cosTheta = cos(gate.theta / 2);
        double sinTheta = sin(gate.theta / 2);

        DTYPE d_r[4];
        if (gate.gname_id == 3)
        {
            d_r[0] = make_cuDoubleComplex(cosTheta, 0.0);
            d_r[1] = make_cuDoubleComplex(0.0, -sinTheta);
            d_r[2] = make_cuDoubleComplex(0.0, -sinTheta);
            d_r[3] = make_cuDoubleComplex(cosTheta, 0.0);
        }
        else if (gate.gname_id == 4)
        {
            d_r[0] = make_cuDoubleComplex(cosTheta, 0.0);
            d_r[1] = make_cuDoubleComplex(-sinTheta, 0.0);
            d_r[2] = make_cuDoubleComplex(sinTheta, 0.0);
            d_r[3] = make_cuDoubleComplex(cosTheta, 0.0);
        }
        else if (gate.gname_id == 5)
        {
            d_r[0] = make_cuDoubleComplex(cosTheta, -sinTheta);
            d_r[1] = make_cuDoubleComplex(0.0, 0.0);
            d_r[2] = make_cuDoubleComplex(0.0, 0.0);
            d_r[3] = make_cuDoubleComplex(cosTheta, sinTheta);
        }

        return d_r;
    }
    else
    {
        switch (gate.gname_id)
        {
        case 1:
            return d_IDE;
        case 2:
            return d_H;
        case 3:
            return d_X;
        case 4:
            return d_Y;
        case 5:
            return d_Z;
        case 6:
            return d_SWAP;
        default:
            return nullptr;
        }
    }
}