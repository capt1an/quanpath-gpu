#pragma once

#include "qgate.h"

#define MAX_DEPTHS 1024 // 假设的最大电路深度
#define MAX_QUBITS 20   // 假设的最大 qubit 数量

class QCircuit
{
public:
    int numQubits;
    int numDepths;
    int numLowQubits;
    QGate gates[MAX_DEPTHS][MAX_QUBITS];
    string name;
    QGateDevice* d_gate_array = nullptr;


    QCircuit();
    QCircuit(int numQubits_, string name_ = "qcircuit");

    //
    // Single-qubit gates
    //
    void h(int qid);
    void x(int qid);
    void y(int qid);
    void z(int qid);
    void rx(double theta, int qid);
    void ry(double theta, int qid);
    void rz(double theta, int qid);

    //
    // 2-qubit gates
    //
    void cx(int ctrl, int targ);
    void cy(int ctrl, int targ);
    void cz(int ctrl, int targ);
    void swap(int qid1, int qid2);

    void copyGatesToDevice();

    //
    // Other operations on quantum circuits
    //
    void barrier();
    void setDepths(int numDepths_);
    void print();
    void printInfo();

    void add_level();
};


int mapNameToID(const char *name);