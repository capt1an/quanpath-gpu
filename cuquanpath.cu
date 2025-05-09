#include "cuquanpath.h"

void QuanPath(QCircuit &qc, Matrix<DTYPE> &hostSv, int numBlocks, int numDepths, int numHighQubits, int numLowQubits)
{
    // 在const memory 初始化门矩阵
    initGateMatricesInConstMemory();

    // 将线路放入GPU的内存空间
    // QGateDevice *d_gate_array = nullptr;
    // copyGatesToDevice(d_gate_array, qc, numLowQubits);
    qc.copyGatesToDevice();

    // 为设备分配状态向量的内存空间
    Matrix<DTYPE> *deviceSv;
    HANDLE_CUDA_ERROR(Matrix<DTYPE>::allocateDeviceMemory(deviceSv, hostSv));

    // auto start = chrono::high_resolution_clock::now();
    Matrix<DTYPE> Opmat = highOMSim(qc, numHighQubits);

    // Step 2. Local SVSim for gates on low-order qubits

    int threadPerBlock = 1 << (numLowQubits - 1);
    assert(threadPerBlock <= 1024);
    int blockPerGrid = (hostSv.row + threadPerBlock - 1) / (threadPerBlock * 2);
    size_t sharedMemSize = (hostSv.row / blockPerGrid) * sizeof(DTYPE);
    assert(sharedMemSize <= 49152);
    SVSim<<<blockPerGrid, threadPerBlock, sharedMemSize>>>(qc.d_gate_array, deviceSv, hostSv.row / blockPerGrid, numDepths, numLowQubits);

    cudaDeviceSynchronize();

    // Step 3. Final merge that requires communication

    Matrix<DTYPE> *ptrOpmat;
    HANDLE_CUDA_ERROR(Matrix<DTYPE>::allocateDeviceMemory(ptrOpmat, Opmat));

    merge<<<blockPerGrid, 128>>>(deviceSv, ptrOpmat);
    cudaDeviceSynchronize();

    HANDLE_CUDA_ERROR(Matrix<DTYPE>::copyDeviceToHost(deviceSv, hostSv));

    // 释放设备内存
    HANDLE_CUDA_ERROR(Matrix<DTYPE>::freeDeviceMemory(deviceSv));
    HANDLE_CUDA_ERROR(Matrix<DTYPE>::freeDeviceMemory(ptrOpmat));

    hostSv.writeToTextFile("sv.txt");
}

/**
 * @brief [TODO] Conduct OMSim for high-order qubits using a thread
 *
 * @param qc a quantum circuit
 * @param numHighQubits the number of high-order qubits
 */
Matrix<DTYPE> highOMSim(QCircuit &qc, int numHighQubits)
{
    int numLowQubits = qc.numQubits - numHighQubits;
    Matrix<DTYPE> opmat, levelmat;
    opmat.identity(1 << numHighQubits);
    levelmat.identity(2);
    for (int j = 0; j < qc.numDepths; ++j)
    {
        int qid = qc.numQubits - 1;

        // get the highest gate matrix
        while (qc.gates[j][qid].isMARK() && qc.gates[j][qid].targetQubits[0] >= numLowQubits)
        {
            // Skip the pseudo placeholder MARK gates placed at control positions
            // when the target gate is applied to high-order qubits
            // If the target gate is applied to low-order qubits, MARK should be regarded as IDE
            --qid;
        }
        // [TODO] Calculate the operation matrix for gates applied to high-order qubits
        // [HINT] We have modified getCompleteMatrix to deal with MARK
        //        In this assignment, MARK is associated with an identity matrix
        // cout << "[TODO] Calculate the operation matrix for gates applied to high-order qubits" << endl;
        // MPI_Abort(MPI_COMM_WORLD, 1);
        levelmat = move(getCompleteMatrix(qc.gates[j][qid]));
        for (int i = qid - 1; i >= numLowQubits; --i)
        {
            if (qc.gates[j][i].isMARK() && qc.gates[j][i].targetQubits[0] >= numLowQubits)
            {
                continue;
            }
            Matrix<DTYPE> tmpmat = move(getCompleteMatrix(qc.gates[j][i]));
            levelmat = move(levelmat.tensorProduct(tmpmat));
        }
        opmat = move(levelmat * opmat);
        // ///////////////////////////////////////////////////////////////////////////
    }
    return opmat;
}

__global__ void SVSim(QGateDevice *d_gate_array, Matrix<DTYPE> *deviceSv, int numStatePerBlock, int numDepths, int numLowQubits)
{
    // int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // printf("idx=%d, blockIdx=%d, threadIdx=%d \n", idx, blockIdx.x, threadIdx.x);

    extern __shared__ DTYPE shared_state[];
    for (int i = threadIdx.x; i < numStatePerBlock; i += blockDim.x)
    {
        shared_state[i] = deviceSv->data[blockIdx.x * numStatePerBlock + i][0];
    }

    __syncthreads();
    
    for (int lev = 0; lev < numDepths; ++lev)
    {
        for (int qid = 0; qid < numLowQubits; ++qid)
        {
            int gate_idx = lev * numLowQubits + qid;
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
        deviceSv->data[blockIdx.x * numStatePerBlock + i][0] = shared_state[i];
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
__global__ void merge(Matrix<DTYPE> *sv, Matrix<DTYPE> *ptrOpmat)
{
    int opmatSize = ptrOpmat->col;
    // const int MAX_OPMAT_SIZE = 32;
    // __shared__ DTYPE sharedOpmat[MAX_OPMAT_SIZE][MAX_OPMAT_SIZE];

    // int row = threadIdx.x / opmatSize;
    // int col = threadIdx.x % opmatSize;
    // if (row < opmatSize && col < opmatSize)
    //     sharedOpmat[row][col] = ptrOpmat->data[row][col];

    // __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int localSvLen = sv->row / opmatSize;
    int totalThreadsNum = blockDim.x * gridDim.x;
    // Calculate the number of elements each thread should process
    int numElementsPerThread = (sv->row + totalThreadsNum - 1) / totalThreadsNum;

    // Calculate the starting index for the current thread
    int startIdx = idx * numElementsPerThread;

    // Loop through the elements this thread is responsible for
    for (int k = 0; k < numElementsPerThread; ++k)
    {
        int currentIdx = startIdx + k;

        if (currentIdx < sv->row)
        {
            DTYPE ans = make_cuDoubleComplex(0, 0);
            for (ll i = 0; i < opmatSize; i++)
            {
                ans = cuCadd(ans, cuCmul(ptrOpmat->data[currentIdx / localSvLen][i], sv->data[currentIdx % localSvLen + localSvLen * i][0]));
            }
            sv->data[currentIdx][0] = ans;
        }
    }
}

__device__ cuDoubleComplex myCmul(cuDoubleComplex a, cuDoubleComplex b)
{
    return make_cuDoubleComplex(
        a.x * b.x - a.y * b.y,
        a.x * b.y + a.y * b.x);
}

__device__ cuDoubleComplex myCadd(cuDoubleComplex a, cuDoubleComplex b)
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

__device__ const DTYPE *get_matrix(QGateDevice &gate)
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