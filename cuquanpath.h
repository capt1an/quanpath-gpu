#pragma once

#include "omsim.h"

/**
 * @brief QuanPath
 *
 * @param qc a quantum circuit
 * @param numWorkers the number of distributed working processes
 * @param myRank the MPI rank of the current process
 */
void QuanPath(QCircuit &qc, Matrix<DTYPE> &hostSv, int numThreads, int numHighQubits, int numLowQubits);

/**
 * @brief [TODO] Conduct OMSim for high-order qubits using a thread
 *
 * @param qc a quantum circuit
 * @param numHighQubits the number of high-order qubits
 */
Matrix<DTYPE> highOMSim(QCircuit &qc, int numHighQubits);

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