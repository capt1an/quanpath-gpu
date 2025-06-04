
#include "utils.h"


CudaUtils::CudaUtils(){

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

}

cudaError_t CudaUtils::startTiming(const std::string& taskName) {
    
    currentTaskName = taskName;
    return HANDLE_CUDA_ERROR(cudaEventRecord(start, 0));
}

cudaError_t CudaUtils::stopTiming()
{
    HANDLE_CUDA_ERROR(cudaEventRecord(stop, 0));
    HANDLE_CUDA_ERROR(cudaEventSynchronize(stop));

    float elapsedTime = 0.0f;
    HANDLE_CUDA_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));

    if (currentTaskName.empty())
        std::cout << "[CUDA] Kernel execution time: " << elapsedTime << " ms" << std::endl;
    else
        std::cout << "[CUDA] " << currentTaskName << " execution time: " << elapsedTime << " ms" << std::endl;

    return cudaSuccess;
}


void CudaUtils::writeDeviceStateVectorToFile(DTYPE *deviceSv, long long stateVectorLen, const char *filename)
{
    // 分配主机内存
    DTYPE *hostSv = (DTYPE *)malloc(stateVectorLen * sizeof(DTYPE));
    if (hostSv == NULL)
    {
        fprintf(stderr, "Host memory allocation failed!\n");
        exit(EXIT_FAILURE);
    }

    // 拷贝 deviceSv 到主机
    HANDLE_CUDA_ERROR(cudaMemcpy(hostSv, deviceSv, stateVectorLen * sizeof(DTYPE), cudaMemcpyDeviceToHost));

    // 打开文件
    FILE *outFile = fopen(filename, "w");
    if (outFile == NULL)
    {
        fprintf(stderr, "Failed to open output file: %s\n", filename);
        free(hostSv);
        exit(EXIT_FAILURE);
    }

    // 写入数据
    for (long long i = 0; i < stateVectorLen; ++i)
    {
        DTYPE val = hostSv[i];
        fprintf(outFile, "%.7f %.7f\n", cuCreal(val), cuCimag(val));
    }

    fclose(outFile);
    printf("State vector written to %s\n", filename);

    // 释放主机内存
    free(hostSv);
}

CudaUtils::~CudaUtils(){
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}
