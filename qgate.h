#pragma once

#include "matrix.h"

class QGate
{
public:
    // 字符串在设备端使用有限制，考虑使用字符数组或整数 ID
    // std::string gname; // gate name
    char gname[32]; // 假设最大门名称长度为 32

    // vector 在设备端不直接支持，需要转换为简单的数组
    // std::vector<int> controlQubits; // the control qubits of the gate
    // std::vector<int> targetQubits; // the target qubits of the gate
    int controlQubits[4]; // 假设最大控制 qubit 数量为 4
    int numControlQubits;
    int targetQubits[4]; // 假设最大目标 qubit 数量为 4
    int numTargetQubits;
    double theta;
    shared_ptr<Matrix<DTYPE>> gmat; // the gate matrix

    QGate();
    QGate(const char *gname_, const int *controls_, int num_controls, const int *targets_, int num_targets);
    QGate(const char *gname_, const int *controls_, int num_controls, const int *targets_, int num_targets, double theta);
    QGate(const QGate &other);

    QGate &operator=(const QGate &other);

    // __host__ __device__ int numQubits() const;   // the number of input/output qubits of the gate
    // __host__ __device__ int numControls() const; // the number of control qubits of the gate
    // __host__ __device__ int numTargets() const;  // the number of target qubits of the gate

    __host__ __device__ int device_strcmp(const char *str1, const char *str2) const
    {
        int i = 0;
        while (str1[i] != '\0' && str2[i] != '\0' && str1[i] == str2[i])
        {
            i++;
        }
        return (unsigned char)str1[i] - (unsigned char)str2[i];
    }

    // Check if the gate is an identity gate
    __host__ __device__ bool isIDE()
    {
        return device_strcmp(gname, "IDE") == 0;
    }

    // Check if the gate is a placeholder gate
    __host__ __device__ bool isMARK() const
    {
        return device_strcmp(gname, "MARK") == 0;
    }

    // Check if the gate is a single-qubit gate
    __host__ __device__ bool isSingle() const
    {
        return device_strcmp(gname, "IDE") != 0 && device_strcmp(gname, "MARK") != 0 && numControlQubits == 0 && numTargetQubits == 1;
    }

    // Check if the gate is a 2-qubit controlled gate
    __host__ __device__ bool is2QubitControlled() const
    {
        return device_strcmp(gname, "MARK") != 0 && numControlQubits == 1 && numTargetQubits == 1;
    }

    // Check if qubit[qid] is a control qubit of the gate
    __host__ __device__ bool isControlQubit(int qid) const
    {
        for (int i = 0; i < numControlQubits; ++i)
        {
            if (controlQubits[i] == qid)
            {
                return true;
            }
        }
        return false;
    }

    // Check if qubit[qid] is a target qubit of the gate
    __host__ __device__ bool isTargetQubit(int qid) const
    {
        if (device_strcmp(gname, "IDE") == 0 || device_strcmp(gname, "MARK") == 0)
            return false;
        for (int i = 0; i < numTargetQubits; ++i)
        {
            if (targetQubits[i] == qid)
            {
                return true;
            }
        }
        return false;
    }

    void print() const; // print the gate information (仅限 host)

    ~QGate();
};


struct QGateDevice {
    int gname_id;

    // int controlQubits[4];
    int numControlQubits;

    // int targetQubits[4];
    int numTargetQubits;

    int numAmps;
    // 步幅
    int strides[4];

    int ctrlmask;

    double theta;
};


//
// Utility functions
//

// Compare two integers by their absolute values
// Control qubits can be negative to denote 0-controlled
bool compareByAbsoluteValue(int a, int b);