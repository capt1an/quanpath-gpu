#include "qcircuit.h"

QCircuit::QCircuit() {}

/**
 * @brief Construct an n-qubit 1-level quantum circuit object
 *
 * @param numQubits_ #Qubits
 * @param numDepths_ #Depths
 */
QCircuit::QCircuit(int numQubits_, string name_)
{
    numQubits = numQubits_;
    numDepths = 0;
    numLowQubits = 0;
    add_level(); // numDepths += 1
    name = name_;
}

//
// Single-qubit gates
//

/**
 * @brief Apply an H gate to qubit[qid]
 *
 * @param qid   qubit id
 */
void QCircuit::h(int qid)
{
    if (!gates[numDepths - 1][qid].isIDE())
    {
        add_level();
    }
    int targets[] = {qid};
    gates[numDepths - 1][qid] = QGate("H", nullptr, 0, targets, 1);
}

/**
 * @brief Apply an X gate to qubit[qid]
 *
 * @param qid   qubit id
 */
void QCircuit::x(int qid)
{
    if (!gates[numDepths - 1][qid].isIDE())
    {
        add_level();
    }
    int targets[] = {qid};
    gates[numDepths - 1][qid] = QGate("X", nullptr, 0, targets, 1);
}

/**
 * @brief Apply a Y gate to qubit[qid]
 *
 */
void QCircuit::y(int qid)
{
    if (!gates[numDepths - 1][qid].isIDE())
    {
        add_level();
    }
    int targets[] = {qid};
    gates[numDepths - 1][qid] = QGate("Y", nullptr, 0, targets, 1);
}

/**
 * @brief Apply a Z gate to qubit[qid]
 *
 * @param qid   qubit id
 */
void QCircuit::z(int qid)
{
    if (!gates[numDepths - 1][qid].isIDE())
    {
        add_level();
    }
    int targets[] = {qid};
    gates[numDepths - 1][qid] = QGate("Z", nullptr, 0, targets, 1);
}

/**
 * @brief Apply an RX gate to qubit[qid]
 *
 * @param param the gate parameter
 * @param qid   qubit id
 */
void QCircuit::rx(double theta, int qid)
{
    if (!gates[numDepths - 1][qid].isIDE())
    {
        add_level();
    }
    int targets[] = {qid};
    gates[numDepths - 1][qid] = QGate("RX", nullptr, 0, targets, 1, theta);
}

/**
 * @brief Apply an RY gate to qubit[qid]
 *
 * @param param the gate parameter
 * @param qid   qubit id
 */
void QCircuit::ry(double theta, int qid)
{
    if (!gates[numDepths - 1][qid].isIDE())
    {
        add_level();
    }
    int targets[] = {qid};
    gates[numDepths - 1][qid] = QGate("RY", nullptr, 0, targets, 1, theta);
}

/**
 * @brief Apply an RZ gate to qubit[qid]
 *
 * @param param the gate parameter
 * @param qid   qubit id
 */
void QCircuit::rz(double theta, int qid)
{
    if (!gates[numDepths - 1][qid].isIDE())
    {
        add_level();
    }
    int targets[] = {qid};
    gates[numDepths - 1][qid] = QGate("RZ", nullptr, 0, targets, 1, theta);
}

//
// 2-qubit gates
//

/**
 * @brief Apply a CX gate to qubit[ctrl] and qubit[targ]
 *
 * @param ctrl  control qubit id
 * @param targ  target qubit id
 */
void QCircuit::cx(int ctrl, int targ)
{
    int start = min(ctrl, targ);
    int end = max(ctrl, targ);
    for (int i = start; i <= end; ++i)
    {
        if (!gates[numDepths - 1][i].isIDE())
        {
            add_level();
            break;
        }
    }
    int controls[] = {ctrl};
    int targets[] = {targ};
    for (int i = start; i <= end; ++i)
    {
        int mark_targets[] = {start, end};
        gates[numDepths - 1][i] = QGate("MARK", controls, 1, mark_targets, 2);
    }
    gates[numDepths - 1][targ] = QGate("CX", controls, 1, targets, 1);
}

/**
 * @brief Apply a CY gate to qubit[ctrl] and qubit[targ]
 *
 * @param ctrl  control qubit id
 * @param targ  target qubit id
 */
void QCircuit::cy(int ctrl, int targ)
{
    int start = min(ctrl, targ);
    int end = max(ctrl, targ);
    for (int i = start; i <= end; ++i)
    {
        if (!gates[numDepths - 1][i].isIDE())
        {
            add_level();
            break;
        }
    }
    int controls[] = {ctrl};
    int targets[] = {targ};
    for (int i = start; i <= end; ++i)
    {
        int mark_targets[] = {start, end};
        gates[numDepths - 1][i] = QGate("MARK", controls, 1, mark_targets, 2);
    }
    gates[numDepths - 1][targ] = QGate("CY", controls, 1, targets, 1);
}

/**
 * @brief Apply a CZ gate to qubit[ctrl] and qubit[targ]
 *
 * @param ctrl  control qubit id
 * @param targ  target qubit id
 */
void QCircuit::cz(int ctrl, int targ)
{
    int start = min(ctrl, targ);
    int end = max(ctrl, targ);
    for (int i = start; i <= end; ++i)
    {
        if (!gates[numDepths - 1][i].isIDE())
        {
            add_level();
            break;
        }
    }
    int controls[] = {ctrl};
    int targets[] = {targ};
    for (int i = start; i <= end; ++i)
    {
        int mark_targets[] = {start, end};
        gates[numDepths - 1][i] = QGate("MARK", controls, 1, mark_targets, 2);
    }
    gates[numDepths - 1][targ] = QGate("CZ", controls, 1, targets, 1);
}

/**
 * @brief Apply a SWAP gate to qubit[qid1] and qubit[qid2]
 *
 * @param qid1  qubit id 1
 * @param qid2  qubit id 2
 */
void QCircuit::swap(int qid1, int qid2)
{
    int start = min(qid1, qid2);
    int end = max(qid1, qid2);
    for (int i = start; i <= end; ++i)
    {
        if (!gates[numDepths - 1][i].isIDE())
        {
            add_level();
            break;
        }
    }
    int targets[] = {start, end};
    for (int i = start; i <= end; ++i)
    {
        gates[numDepths - 1][i] = QGate("MARK", nullptr, 0, targets, 2);
    }
    gates[numDepths - 1][end] = QGate("SWAP", nullptr, 0, targets, 2);
}

void QCircuit::copyGatesToDevice()
{
    int totalGates = numDepths * numLowQubits;
    QGateDevice *h_gate_array = new QGateDevice[totalGates];

    for (int depth = 0; depth < numDepths; ++depth)
    {
        for (int qid = 0; qid < numLowQubits; ++qid)
        {
            const QGate &src = gates[depth][qid];
            QGateDevice &dst = h_gate_array[depth * numLowQubits + qid];

            dst.gname_id = mapNameToID(src.gname);
            dst.numAmps = 1 << src.numTargetQubits;

            dst.numControlQubits = src.numControlQubits;
            dst.numTargetQubits = src.numTargetQubits;
            dst.theta = src.theta;

            for (ll idx = 0; idx < dst.numAmps; ++idx)
            {
                ll stride = 0;
                for (int j = 0; j < src.numTargetQubits; ++j)
                {
                    if (idx & (1 << j))
                    { // if the j-th bit of idx is 1
                        stride += (1 << src.targetQubits[j]);
                    }
                }
                dst.strides[idx] = stride;
            }

            if (src.controlQubits == 0)
                dst.ctrlmask = 0;
            else
                dst.ctrlmask = 1 << src.controlQubits[0];
        }
    }

    // 在 GPU 上分配内存
    cudaMalloc(&d_gate_array, sizeof(QGateDevice) * totalGates);

    // 拷贝到 GPU
    cudaMemcpy(d_gate_array, h_gate_array, sizeof(QGateDevice) * totalGates, cudaMemcpyHostToDevice);

    // 清理 host 临时缓冲区
    delete[] h_gate_array;
}

/**
 * @brief Add a barrier to the quantum circuit
 */
void QCircuit::barrier()
{
    add_level();
}

/**
 * @brief Set the circuit depth to numDepths_
 *
 * @param numDepths_  the target circuit depth
 */
void QCircuit::setDepths(int numDepths_)
{
    int range = numDepths_ - numDepths;
    for (int i = 0; i < range; ++i)
    {
        add_level();
    }
}

/**
 * @brief Print the structure of the quantum circuit
 */
void QCircuit::print()
{
    printInfo();
    int start = 0;
    if (numQubits >= 6)
    {
        start = numQubits - 6;
    }
    for (int i = numQubits - 1; i >= start; --i)
    {
        cout << "q[" << i << "]\t";
        for (int j = 0; j < numDepths; ++j)
        {
            if (j > 10)
            {
                cout << "...";
                break;
            }
            if (gates[j][i].isControlQubit(i))
            {
                cout << "C";
            }
            else if (gates[j][i].isTargetQubit(i))
            {
                cout << "T";
            }
            cout << gates[j][i].gname << "\t";
        }
        cout << endl;
    }
}

/**
 * @brief Print the information of the quantum circuit
 */
void QCircuit::printInfo()
{
    cout << "[INFO] [" << name << "] numQubits: [" << numQubits << "] numDepths: [" << numDepths << "]" << endl;
}

/**
 * @brief Add a new level full of IDE gates to the circuit
 */
void QCircuit::add_level()
{
    if (numDepths >= MAX_DEPTHS)
    {
        std::cerr << "[ERROR] Maximum circuit depth reached (" << MAX_DEPTHS << ")" << std::endl;
        return;
    }
    for (int i = 0; i < numQubits; ++i)
    {
        int targets[] = {i};
        gates[numDepths][i] = QGate("IDE", nullptr, 0, targets, 1);
    }
    numDepths++;
}

int mapNameToID(const char *name)
{
    if (strcmp(name, "IDE") == 0 || strcmp(name, "MARK") == 0)
        return 1;
    if (strcmp(name, "H") == 0)
        return 2;
    if (strcmp(name, "X") == 0 || strcmp(name, "CX") == 0 || strcmp(name, "RX") == 0)
        return 3;
    if (strcmp(name, "Y") == 0 || strcmp(name, "CY") == 0 || strcmp(name, "RY") == 0)
        return 4;
    if (strcmp(name, "Z") == 0 || strcmp(name, "CZ") == 0 || strcmp(name, "RZ") == 0)
        return 5;
    if (strcmp(name, "SWAP") == 0 || strcmp(name, "CSWAP") == 0)
        return 6;
    return 0;
}