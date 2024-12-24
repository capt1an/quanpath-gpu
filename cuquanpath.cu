#include "cuquanpath.h"

void QuanPath(QCircuit &qc, Matrix<DTYPE> &hostSv, int numBlocks, int numHighQubits, int numLowQubits)
{

    // 为设备分配状态向量的内存空间
    Matrix<DTYPE> *deviceSv;
    HANDLE_CUDA_ERROR(Matrix<DTYPE>::allocateDeviceMemory(deviceSv, hostSv));

    // auto start = chrono::high_resolution_clock::now();
    // Step 1. Calculate the high-order operation matrix in cpu
    Matrix<DTYPE> Opmat = highOMSim(qc, numHighQubits);

    // Step 2. Local SVSim for gates on low-order qubits
    Matrix<DTYPE> *deviceGm;
    Matrix<DTYPE> gateMatrix = Matrix<DTYPE>(4, 4);
    HANDLE_CUDA_ERROR(Matrix<DTYPE>::allocateDeviceMemory(deviceGm, gateMatrix));

    // int threadPerBlock = 256;
    // int blockPerGrid = (hostSv.row + threadPerBlock - 1) / threadPerBlock;
    int threadPerBlock = (hostSv.row + numBlocks - 1) / numBlocks;
    int blockPerGrid = numBlocks;
    threadPerBlock = threadPerBlock < 16 ? 16 : threadPerBlock;
    threadPerBlock = threadPerBlock > 1024 ? 1024 : threadPerBlock;

    for (int lev = 0; lev < qc.numDepths; ++lev)
    {
        for (int qid = 0; qid < numLowQubits; ++qid)
        {
            QGate &gate = qc.gates[lev][qid];
            if (gate.isIDE() || gate.isMARK())
            {
                continue;
            }
            gateMatrix = getCompleteMatrix(gate);
            HANDLE_CUDA_ERROR(Matrix<DTYPE>::copyHostToDevice(gateMatrix, deviceGm));
            if (gate.isSingle())
                SVSimForSingleQubit<<<blockPerGrid, threadPerBlock>>>(deviceGm, numLowQubits, deviceSv, gate.targetQubits[0]);
            else if (gate.numControls() != 0)
            {
                int q0 = gate.controlQubits[0], q1 = gate.targetQubits[0];
                SVSimForTwoQubit<<<blockPerGrid, threadPerBlock>>>(deviceGm, numLowQubits, deviceSv, min(q0, q1), max(q0, q1));
            }
            else
            {
                int q0 = gate.targetQubits[0], q1 = gate.targetQubits[1];
                SVSimForTwoQubit<<<blockPerGrid, threadPerBlock>>>(deviceGm, numLowQubits, deviceSv, min(q0, q1), max(q0, q1));
            }
        }
    }

    cudaDeviceSynchronize();

    // HANDLE_CUDA_ERROR(Matrix<DTYPE>::copyDeviceToHost(deviceSv, hostSv));

    // 释放设备内存
    // HANDLE_CUDA_ERROR(Matrix<DTYPE>::freeDeviceMemory(deviceSv));
    HANDLE_CUDA_ERROR(Matrix<DTYPE>::freeDeviceMemory(deviceGm));

    // Step 3. Final merge that requires communication
    // dim3 mergegrid(1);
    // dim3 mergeblock(hostSv.row / numThreads, numThreads);

    Matrix<DTYPE> *ptrOpmat;
    // HANDLE_CUDA_ERROR(Matrix<DTYPE>::allocateDeviceMemory(deviceSv, hostSv));
    HANDLE_CUDA_ERROR(Matrix<DTYPE>::allocateDeviceMemory(ptrOpmat, Opmat));

    merge<<<blockPerGrid, 128>>>(deviceSv, ptrOpmat);
    // auto end = chrono::high_resolution_clock::now();
    cudaDeviceSynchronize();

    HANDLE_CUDA_ERROR(Matrix<DTYPE>::copyDeviceToHost(deviceSv, hostSv));

    HANDLE_CUDA_ERROR(Matrix<DTYPE>::freeDeviceMemory(deviceSv));
    HANDLE_CUDA_ERROR(Matrix<DTYPE>::freeDeviceMemory(ptrOpmat));

    // chrono::duration<double> duration = end - start;
    // cout << "Svsim simulation completed in " << duration.count() << " seconds." << endl;
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

/**
 * @brief Conduct SVSim for gate on single qubit
 *
 * @param localSv the local state vector pointer
 * @param gateMatrix the gate matrix pointer
 * @param numLowQubits the number of low-order qubits
 * @param qidx the index of target qubit
 */
// __global__ void SVSimForSingleQubit(Matrix<DTYPE> *gateMatrix, int numLowQubits, Matrix<DTYPE> *localSv, int qidx)
// {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     int i = (idx / (1 << qidx)) * (1 << (qidx + 1));
//     int j = idx % (1 << qidx);
//     int p = i | j;
//     // 将 gateMatrix 加载到共享内存
//     __shared__ DTYPE sharedGateMatrix[2][2];
//     if (threadIdx.x < 4)
//     {
//         int row = threadIdx.x / 2;
//         int col = threadIdx.x % 2;
//         sharedGateMatrix[row][col] = gateMatrix->data[row][col];
//     }
//     __shared__ DTYPE sharedLocalSv[2048];
//     sharedLocalSv[p % 2048] = localSv->data[p][0];
//     sharedLocalSv[(p | 1 << qidx) % 2048] = localSv->data[p | 1 << qidx][0];
//     __syncthreads();

//     if (idx < localSv->row / 2)
//     {
//         // int i = (idx / (1 << qidx)) * (1 << (qidx + 1));
//         // int j = idx % (1 << qidx);
//         // int p = i | j;
//         DTYPE q0 = sharedLocalSv[p % 2048];
//         DTYPE q1 = sharedLocalSv[(p | 1 << qidx) % 2048];
//         sharedLocalSv[p % 2048] = cuCadd(cuCmul(sharedGateMatrix[0][0], q0), cuCmul(sharedGateMatrix[0][1], q1));
//         sharedLocalSv[(p | 1 << qidx) % 2048] = cuCadd(cuCmul(sharedGateMatrix[1][0], q0), cuCmul(sharedGateMatrix[1][1], q1));
//     }

//     localSv->data[p][0] = sharedLocalSv[p % 2048];
//     localSv->data[p | 1 << qidx][0] = sharedLocalSv[(p | 1 << qidx) % 2048];
// }

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