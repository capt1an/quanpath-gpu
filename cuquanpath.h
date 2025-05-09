#pragma once

#include "omsim.h"

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
void QuanPath(QCircuit &qc, Matrix<DTYPE> &hostSv, int numThreads, int numDepths, int numHighQubits, int numLowQubits);

/**
 * @brief [TODO] Conduct OMSim for high-order qubits using a thread
 *
 * @param qc a quantum circuit
 * @param numHighQubits the number of high-order qubits
 */
Matrix<DTYPE> highOMSim(QCircuit &qc, int numHighQubits);

__global__ void SVSim(QGateDevice *d_gate_array, Matrix<DTYPE> *deviceSv, int numStatePerBlock, int numDepths, int numLowQubits);


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
__global__ void merge(Matrix<DTYPE> *sv, Matrix<DTYPE> *ptrOpmat);



__device__ cuDoubleComplex myCmul(cuDoubleComplex a, cuDoubleComplex b);

__device__ cuDoubleComplex myCadd(cuDoubleComplex a, cuDoubleComplex b);

cudaError_t initGateMatricesInConstMemory();

cudaError_t copyGatesToDevice(QGateDevice* d_gate_array, QCircuit& qc, int numLowQubits);

__device__ const DTYPE *get_matrix(QGateDevice &gate);

__device__ bool isLegalControlPattern(int qid, QGateDevice &gate);