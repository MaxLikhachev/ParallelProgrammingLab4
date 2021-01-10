#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <omp.h>
#include <cmath>

using namespace std;

#define BLOCK_SIZE 64
#define N 32

#define EPS 0.01;
#define CRITICAL_COUNT 100;

void display(int matrixSizeDepth, int matrixSizeWidth, float* matrix)
{
    for (int i = 0; i < matrixSizeDepth; i++)
    {
        for (int j = 0; j < matrixSizeWidth; j++)
            cout << matrix[i * matrixSizeDepth + j] << " ";
        cout << endl;
    }
}

void display(int vectorSize, float* vector)
{
    for (int i = 0; i < vectorSize; i++)
        cout << vector[i] << " ";
    cout << endl;
}

void initRandom(int vectorSize, float* vector)
{
    for (int i = 0; i < vectorSize; i++)
        vector[i] = 1 + rand() % 10 * 1.0;
}

void initRandom(int matrixSizeDepth, int matrixSizeWidth, float* matrix)
{
    for (int i = 0; i < matrixSizeDepth; i++)
        for (int j = 0; j < matrixSizeWidth; j++)
        {
            matrix[i * matrixSizeDepth + j] = 1 + rand() % 10 * 1.0;
            if (i == j) matrix[i * matrixSizeDepth + j] *= matrixSizeDepth;
        }
}

void initNull(int vectorSize, float* vector)
{
    for (int i = 0; i < vectorSize; i++)
        vector[i] = 0.0;
}

void initNull(int matrixSizeDepth, int matrixSizeWidth, float* matrix)
{
    for (int i = 0; i < matrixSizeDepth; i++)
        for (int j = 0; j < matrixSizeWidth; j++)
            matrix[i * matrixSizeDepth + j] = 0.0;
}

bool sequentialIsComplete(int vectorSize, float* prev, float* curr)
{
    bool flag = false;
    float error = 0.0, eps = EPS;
    for (int i = 0; i < vectorSize; i++)
        error += abs(curr[i] - prev[i]);
    if (error < eps)
    {
        cout << "Finished successfully\n";
        flag = true;
    }
    return flag;
}

void sequentialMultiMatrixVector(int vectorSize, float* matrix, float* vector, float* result)
{
    for (int i = 0; i < vectorSize; i++)
        for (int j = 0; j < vectorSize; j++)
            result[i] += matrix[i * vectorSize + j] * vector[j];
}

void sequentialSubVectors(int vectorSize, float* vectorL, float* vectorR, float* result)
{
    for (int i = 0; i < vectorSize; i++)
        result[i] = vectorL[i] - vectorR[i];
}

void sequentialCopyVectors(int vectorSize, float* vector, float* result)
{
    for (int i = 0; i < vectorSize; i++)
        result[i] = vector[i];
}

void parallelOpenMPMultiMatrixVector(int vectorSize, float* matrix, float* vector, float* result)
{
#pragma omp parallel for
    for (int i = 0; i < vectorSize; i++)
    {
#pragma omp parallel for
        for (int j = 0; j < vectorSize; j++)
            result[i] += matrix[i * vectorSize + j] * vector[j];
    }
}

void parallelOpenMPSubVectors(int vectorSize, float* vectorL, float* vectorR, float* result)
{
#pragma omp parallel for
    for (int i = 0; i < vectorSize; i++)
        result[i] = vectorL[i] - vectorR[i];
}

void parallelOpenMPCopyVectors(int vectorSize, float* vector, float* result)
{
#pragma omp parallel for
    for (int i = 0; i < vectorSize; i++)
        result[i] = vector[i];
}

bool parallelOpenMPIsComplete(int vectorSize, float* prev, float* curr)
{
    bool flag = false;
    float error = 0.0, eps = EPS;
#pragma omp parallel for
    for (int i = 0; i < vectorSize; i++)
        error += abs(curr[i] - prev[i]);
    if (error < eps)
    {
        cout << "Finished successfully\n";
        flag = true;
    }
    return flag;
}

bool parallelCudaIsComplete(int vectorSize, float* prev, float* curr)
{
    bool flag = false;
    float error = 0.0, eps = EPS;
    for (int i = 0; i < vectorSize; i++)
        error += abs(curr[i] - prev[i]);
    if (error < eps)
    {
        cout << "Finished successfully\n";
        flag = true;
    }
    return flag;
}

__global__ void parallelCudaMultiMatrixVectorKernel(float* matrix, float* vector, float* result)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float temp = 0;
    if (i < N)
        for (int j = 0; j < N; j++)
            temp += matrix[i * N + j] * vector[j];
    result[i] = temp;
}

__global__ void parallelCudaSubVectorsKernel(float* vectorL, float* vectorR, float* result)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    result[i] = vectorL[i] - vectorR[i];
}

__global__ void parallelCudaCopyVectorsKernel(float* vector, float* result)
{
    int i = threadIdx.x;
    result[i] = vector[i];
}

void initAlpha(int matrixSize, float* matrix, float* result)
{
    initNull(matrixSize, matrixSize, result);
    for (int i = 0; i < matrixSize; i++)
        for (int j = 0; j < matrixSize; j++)
            if (i == j) 
                result[i * matrixSize + i] = 0.0;
            else
                result[i * matrixSize + j] = (double) matrix[i * matrixSize + j] / (double)matrix[i * matrixSize + i];         
}

void initBeta(int vectorSize, float* matrix, float* vector, float* result)
{
    initNull(vectorSize, result);
    for (int i = 0; i < vectorSize; i++)
        result[i] = vector[i] / matrix[i * vectorSize + i];
}

void init(int matrixSize, float* matrix, float* basis, float *prev, float* curr, float* alpha, float* beta)
{
    initRandom(matrixSize, curr);
    initRandom(matrixSize, matrixSize, matrix);

    sequentialMultiMatrixVector(matrixSize, matrix, curr, basis);

    initAlpha(matrixSize, matrix, alpha);
    initBeta(matrixSize, matrix, basis, beta);

    initNull(matrixSize, prev);
    initNull(matrixSize, curr);
}

void parallelOpenMPCalculate(int vectorSize, float* alpha, float* beta, float* prev, float* curr)
{
    cudaEvent_t start, stop;
    float KernelTime;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    int count = 0, criticalCount = CRITICAL_COUNT;
    for (count = 0; count < criticalCount; count++)
    {
        parallelOpenMPMultiMatrixVector(vectorSize, alpha, prev, curr);
        parallelOpenMPSubVectors(vectorSize, beta, curr, curr);
        if (parallelOpenMPIsComplete(vectorSize, prev, curr)) break;
        parallelOpenMPCopyVectors(vectorSize, curr, prev);
    }

    cudaThreadSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&KernelTime, start, stop);
    cout << "Parallel OpenMP calculate iterations:" << count << " by time: " << KernelTime * 1000 << " mcs\n";
}

void sequentialCalculate(int vectorSize, float* alpha, float* beta, float* prev, float* curr)
{
    cudaEvent_t start, stop;
    float KernelTime;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    int count = 0, criticalCount = CRITICAL_COUNT;
    for (count = 0; count < criticalCount; count++)
    {
        sequentialMultiMatrixVector(vectorSize, alpha, prev, curr);
        sequentialSubVectors(vectorSize, beta, curr, curr);
        if (sequentialIsComplete(vectorSize, prev, curr)) break;
        sequentialCopyVectors(vectorSize, curr, prev);
    }

    cudaThreadSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&KernelTime, start, stop);
    cout << "Sequential calculate iterations:" << count <<" by time: " << KernelTime * 1000 << " mcs\n";
}

cudaError_t parallelCudaCalculate(int vectorSize, float* alpha, float* beta, float* prev, float* curr)
{
    float* dev_alpha;
    float* dev_beta;
    float* dev_prev;
    float* dev_curr;
    
    cudaError_t cudaStatus;
    cudaEvent_t start, stop;
    float KernelTime;
    
    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for matrix.
    cudaStatus = cudaMalloc((void**)&dev_alpha, (N * N) * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc (dev_alpha) failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_beta, N * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc (dev_beta) failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_prev, N * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc (dev_prev) failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_curr, N * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc (dev_curr) failed!");
        goto Error;
    }


    // Copy input matrixes from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_alpha, alpha, (N * N) * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy (dev_alpha, alpha) failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_beta, beta, N * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy (dev_beta, beta) failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_prev, prev, N * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy (dev_prev, prev) failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_curr, curr, N * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy (dev_curr, curr) failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    int blockSize, gridSize;
    // Number of threads in each thread block
    blockSize = 1024;
    // Number of thread blocks in grid
    gridSize = (int)ceil((float)N / blockSize);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // Global memory
    int count = 0, criticalCount = CRITICAL_COUNT;
    for (count = 0; count < criticalCount; count++)
    {
        // parallelCudaMultiMatrixVectorKernel << <gridSize, blockSize >> > (alpha, prev, curr);
        // Check for any errors launching the kernel
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "parallelCudaMultiMatrixVectorKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            goto Error;
        }

        // cudaDeviceSynchronize waits for the kernel to finish, and returns any errors encountered during the launch.
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching parallelCudaMultiMatrixVectorKernel!\n", cudaStatus);
            goto Error;
        }
        parallelCudaSubVectorsKernel << <gridSize, blockSize >> > (beta, curr, curr);
        // Check for any errors launching the kernel
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "parallelCudaSubVectorsKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            goto Error;
        }

        // cudaDeviceSynchronize waits for the kernel to finish, and returns
        // any errors encountered during the launch.
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching parallelCudaSubVectorsKernel!\n", cudaStatus);
            goto Error;
        }

        if (parallelCudaIsComplete(vectorSize, prev, curr)) break;

        // Check for any errors launching the kernel
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "parallelCudaIsComplete launch failed: %s\n", cudaGetErrorString(cudaStatus));
            goto Error;
        }

        // cudaDeviceSynchronize waits for the kernel to finish, and returns any errors encountered during the launch.
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching parallelCudaIsComplete!\n", cudaStatus);
            goto Error;
        }

        parallelCudaCopyVectorsKernel << <gridSize, blockSize >> > (curr, prev);
        // Check for any errors launching the kernel
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "parallelCudaCopyVectorsKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            goto Error;
        }

        // cudaDeviceSynchronize waits for the kernel to finish, and returns
        // any errors encountered during the launch.
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching parallelCudaCopyVectorsKernel!\n", cudaStatus);
            goto Error;
        }
    }
    cout << "Finished with iterations count: " << count << endl;

    cudaThreadSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&KernelTime, start, stop);
    cout << "Sequential calculate iterations:" << count << " by time: " << KernelTime * 1000 << " mcs\n";

    // Copy output array from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(alpha, dev_alpha, (N * N) * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy (alpha, dev_alpha) failed!");
        goto Error;
    }
    cudaStatus = cudaMemcpy(beta, dev_beta, N * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy (beta, dev_beta) failed!");
        goto Error;
    }
    cudaStatus = cudaMemcpy(curr, dev_curr, N * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy (curr, dev_curr) failed!");
        goto Error;
    }
    cudaStatus = cudaMemcpy(prev, dev_prev, N * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy (prev, dev_prev) failed!");
        goto Error;
    }

Error:
    cudaFree(dev_alpha);
    cudaFree(dev_beta);
    cudaFree(dev_prev);
    cudaFree(dev_curr);

    return cudaStatus;
}

int main()
{
    srand(time(NULL));

    cout << "Enter matrix size: ";
    int matrixSize = 0;
    cin >> matrixSize;

    float* matrix = new float[matrixSize * matrixSize];
    float* basis = new float[matrixSize];
    float* curr = new float[matrixSize];
    float* prev = new float[matrixSize];

    float* alpha = new float[matrixSize * matrixSize];
    float* beta = new float[matrixSize];

    init(matrixSize, matrix, basis, prev, curr, alpha, beta);

    // <-- Sequential method
    sequentialCalculate(matrixSize, alpha, beta, prev, curr);
    // -- >

    initNull(matrixSize, prev);
    initNull(matrixSize, curr);
    
    // <-- Parallel OpenMP method
    parallelOpenMPCalculate(matrixSize, alpha, beta, prev, curr);
    // -- >

    initNull(matrixSize, prev);
    initNull(matrixSize, curr);

    // <-- Parallel CUDA method
    cudaError_t cudaStatus = parallelCudaCalculate(matrixSize, alpha, beta, prev, curr);

    if (cudaStatus != cudaSuccess) {
        cout << "calculateWithCuda failed!\n";
        return 1;
    }

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        cout << "cudaDeviceReset failed!\n";
        return 1;
    }
    // -- >

    return 0;
}
