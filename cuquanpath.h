#pragma once

#include "qcircuit.h"
#include "utils.h"

#define MAX_SHARED_MEMORY 49152
#define MAX_THREAD_PER_BLOCK 1024
// 在常量内存中定义门矩阵
/*映射关系
"IDE" 	1
"H" 	2
"X" 	3
"Y" 	4
"Z" 	5
"SWAP" 	6
*/
__constant__ DTYPE d_IDE[4];     // 2x2
__constant__ DTYPE d_H[4];       // 2x2
__constant__ DTYPE d_X[4];       // 2x2
__constant__ DTYPE d_Y[4];       // 2x2
__constant__ DTYPE d_Z[4];       // 2x2
__constant__ DTYPE d_SWAP[16];   // 4x4

/**
 * @brief QuanPath
 *
 * @param qc a quantum circuit
 * @param numWorkers the number of distributed working processes
 * @param myRank the MPI rank of the current process
 */
void QuanPath(QCircuit &qc);

/**
 * @brief [TODO] Conduct OMSim for high-order qubits using a thread
 *
 * @param qc a quantum circuit
 * @param numHighQubits the number of high-order qubits
 */
// Matrix<DTYPE> highOMSim(QCircuit &qc, int numHighQubits);

__global__ void highTensorProduct(QGateDevice *d_gate_array,DTYPE *ptrOpmat, int numDepths, int numQubits,int numHighQubits);

__global__ void SVSim(QGateDevice *d_gate_array, DTYPE *deviceSv, int numStatePerBlock, int numDepths, int numQubits, int numLowQubits);

/**
 * @brief Conduct SVSim for gate on single qubit
 *
 * @param localSv the local state vector pointer
 * @param gateMatrix the gate matrix pointer
 * @param numLowQubits the number of low-order qubits
 * @param qidx the index of target qubit
 */
__global__ void SVSimForSingleQubit(Matrix<DTYPE> *gateMatrix, int numLowQubits, Matrix<DTYPE> *localSv, int qidx);

/**
 * @brief Conduct SVSim for gate on two qubits
 *
 * @param localSv the local state vector pointer
 * @param gateMatrix the gate matrix pointer
 * @param numLowQubits the number of low-order qubits
 * @param qlow low index of target qubit
 * @param qhigh high index of target qubit
 */
__global__ void SVSimForTwoQubit(Matrix<DTYPE> *gateMatrix, int numLowQubits, Matrix<DTYPE> *localSv, int qlow, int qhigh);

/**
 * @brief [TODO] Conduct the final merge operation in QuanPath
 *
 * @param sv the state vector
 * @param ptrOpmat the pointer to the high-order operation matrix
 */
__global__ void merge(DTYPE *deviceSv, DTYPE *ptrOpmat, int svLen, int highMatrixSize, int localSvLen, int numElementsPerThread);

cudaError_t initGateMatricesInConstMemory();

cudaError_t copyGatesToDevice(QGateDevice* d_gate_array, QCircuit& qc, int numLowQubits);

__device__ __forceinline__ cuDoubleComplex myCmul(cuDoubleComplex a, cuDoubleComplex b);

__device__ __forceinline__ cuDoubleComplex myCadd(cuDoubleComplex a, cuDoubleComplex b);

__device__ __forceinline__ const DTYPE *get_matrix(QGateDevice &gate);

__device__ __forceinline__ bool isLegalControlPattern(int qid, QGateDevice &gate);