#include "omsim.h"

/**
 * @brief [TODO] Conduct operation matrix simulation of a quantum circuit
 * 
 * @param sv the state vector
 * @param qc a quantum circuit
 * @return Matrix<DTYPE> the operation matrix
 */
Matrix<DTYPE> OMSim(Matrix<DTYPE>& sv, QCircuit& qc) {
    Matrix<DTYPE> opmat, levelmat;
    opmat.identity(sv.row);
    levelmat.identity(2);

    // calculate the operation matrix of the quantum circuit
    for (int j = 0; j < qc.numDepths; ++ j) {
        int qid = qc.numQubits-1;

        // get the highest gate matrix
        while (qc.gates[j][qid].isMARK()) {
            // skip the pseudo placeholder MARK gates placed at control positions
            -- qid;
        }
        if (qid < 0) {
            cout << "[ERROR] Invalid level with no target gate: " << j << endl;
            exit(1);
        }
        // [TODO] Calculate the operation matrix of level j //////////////////////////
        // cout << "[TODO] Calculate the operation matrix of level j" << endl;
        // exit(1);
        // [TODO] Step 1. Let levelmat be the complete gate matrix of the highest gate
        levelmat = move(getCompleteMatrix(qc.gates[j][qid]));
        // ///////////////////////////////////////////////////////////////////////////
        // [TODO] Step 2. Get the complete gate matrices of the remaining gates
        //        Step 2.1. Skip the MARK gates
        //        Step 2.2. Calculate the tensor product of the gate matrices
        for (int i = qid - 1; i >= 0; -- i) {
            if (qc.gates[j][i].isMARK()) {
                continue;
            }
            Matrix<DTYPE> tmpmat = move(getCompleteMatrix(qc.gates[j][i]));
            levelmat = move(levelmat.tensorProduct(tmpmat));
        }
        // ///////////////////////////////////////////////////////////////////////////
        // [TODO] Step 3. Update the operation matrix opmat for the entire circuit
        opmat = levelmat * opmat;
        // ///////////////////////////////////////////////////////////////////////////
    }
    // update the state vector sv
    sv = opmat * sv;
    return opmat;
}

//
// Utility functions
//

/**
 * @brief [TODO] Get a complete gate matrix according to the applied qubits
 * 
 * @param gate the processing gate
 * @return Matrix<DTYPE> a complete gate matrix
 */
Matrix<DTYPE> getCompleteMatrix(QGate& gate) {
    if (gate.isMARK() || gate.isIDE() || gate.isSingle()) {
        return * gate.gmat;
    }
    if (gate.is2QubitControlled()) {
        // [TODO] Return the complete matrix of a 2-qubit controlled gate
        // cout << "[TODO] Return the complete matrix of a 2-qubit controlled gate" << endl;
        // exit(1);
        return genControlledGateMatrix(gate);
        // ///////////////////////////////////////////////////////////////////////////
    }
    if (gate.gname == "SWAP") {
        // [TODO] Return the complete matrix of a SWAP gate
        // cout << "[TODO] Return the complete matrix of a SWAP gate" << endl;
        // exit(1);
        return genSwapGateMatrix(gate);
        // ///////////////////////////////////////////////////////////////////////////
    }
    cout << "[ERROR] getCompleteMatrix: " << gate.gname << " not implemented" << endl;
    exit(1);
}

/**
 * @brief [TODO] Generate the gate matrix of a controlled gate
 *
 * @param gate the processing gate
 * @return Matrix<DTYPE> a complete gate matrix
 */
Matrix<DTYPE> genControlledGateMatrix(QGate& gate) {
    int ctrl = gate.controlQubits[0];
    int targ = gate.targetQubits[0];
    Matrix<DTYPE> ctrlmat, basismat, IDE;
    ctrlmat.zero(1 << (abs(ctrl - targ) + 1), 1 << (abs(ctrl - targ) + 1)); // initialize the complete matrix with all zeros
    IDE.identity(2);
    ll mask = ctrl > targ ? (1 << (ctrl - targ - 1)) : 1; // mask the control qubit

    for (ll i = 0; i < (1 << abs(ctrl-targ)); ++ i) {
        // basismat = | i >< i |
        basismat.zero(1 << abs(ctrl-targ), 1 << abs(ctrl-targ));
        basismat.data[i][i] = make_cuDoubleComplex(1.0,0.0);

        // [TODO] Calculate the complete gate matrix of a 2-qubit controlled gate
        // [HINT] Case 1. If ctrl = 1 and ctrl > targ, ctrlmat += | i >< i | \otimes gate
        //        Case 2. If ctrl = 1 and ctrl < targ, ctrlmat += gate \otimes | i >< i |
        //        Case 3. If ctrl = 0 and ctrl > targ, ctrlmat += | i >< i | \otimes IDE
        //        Case 4. If ctrl = 0 and ctrl < targ, ctrlmat += IDE \otimes | i >< i |
        // cout << "[TODO] Calculate the complete gate matrix of a 2-qubit controlled gate" << endl;
        // exit(1);
        if ((i & mask) == mask) { // control qubit = 1
            if (ctrl > targ) { // ctrlmat += | i >< i | \otimes gate
                ctrlmat += basismat.tensorProduct(*gate.gmat);
            } else { // ctrlmat += gate \otimes | i >< i |
                ctrlmat += gate.gmat->tensorProduct(basismat);
            }
        } else {
            if (ctrl > targ) { // ctrlmat += | i >< i | \otimes IDE
                ctrlmat += basismat.tensorProduct(IDE);
            } else { // ctrlmat += IDE \otimes | i >< i |
                ctrlmat += IDE.tensorProduct(basismat);
            }
        }
        // ///////////////////////////////////////////////////////////////////////////
    }
    return ctrlmat;
}

/**
 * @brief Generate the gate matrix of a SWAP gate
 * 
 * @param gate the processing SWAP gate
 * @return Matrix<DTYPE> a complete gate matrix
 */
Matrix<DTYPE> genSwapGateMatrix(QGate& gate) {
    // when adding a SWAP, the target qubits are sorted in ascending order
    int span = gate.targetQubits[1] - gate.targetQubits[0] + 1;

    Matrix<DTYPE> mat;
    mat.identity(1 << span);

    ll mask0 = (1 << (span - 1));
    ll mask1 = 1;
    ll row;

    for (ll i = 0; i < (1 << span); ++ i) {
        if ((i & mask0) == 0 && (i & mask1) == mask1) {
            // i   := |0..1>
            // row := |1..0>
            row = i ^ mask0 ^ mask1;
            swapRow(i, row, mat);
        }
    }
    return mat;
}

/**
 * @brief Swap two rows of a gate matrix
 * 
 * @param r1   row index 1
 * @param r2   row index 2
 * @param gate return value
 */
void swapRow(ll r1, ll r2, Matrix<DTYPE>& gate) {
    vector<DTYPE> tmp(gate.row);

    copy(gate.data[r1], gate.data[r1] + gate.row, tmp.begin());
    copy(gate.data[r2], gate.data[r2] + gate.row, gate.data[r1]);
    copy(tmp.begin(), tmp.end(), gate.data[r2]);
}