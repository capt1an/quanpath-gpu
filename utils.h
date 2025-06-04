#pragma once

#include "qcircuit.h"


class CudaUtils
{
    cudaEvent_t start, stop;
    std::string currentTaskName;

public:
    CudaUtils();
    ~CudaUtils();

    cudaError_t startTiming(const std::string &taskName = "");
    cudaError_t stopTiming();
    void writeDeviceStateVectorToFile(DTYPE *deviceSv, long long stateVectorLen, const char *filename = "sv.txt");
};
