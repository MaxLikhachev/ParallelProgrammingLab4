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

#define EPS 0.1;
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
        flag = true;
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

__global__ void parallelCudaIsComplete(float* prev, float* curr, bool *flag)
{
    float error = 0.0, eps = EPS;
    for (int i = 0; i < N; i++)
        error += abs(curr[i] - prev[i]);
    if (error < eps)
        *flag = true;
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
    if (i < N)
        result[i] = vectorL[i] - vectorR[i];
}

__global__ void parallelCudaCopyVectorsKernel(float* vector, float* result)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
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
    int count = 0, criticalCount = CRITICAL_COUNT;
    for (count = 0; count < criticalCount; count++)
    {
        parallelOpenMPMultiMatrixVector(vectorSize, alpha, prev, curr);
        parallelOpenMPSubVectors(vectorSize, beta, curr, curr);
        if (parallelOpenMPIsComplete(vectorSize, prev, curr)) break;
        parallelOpenMPCopyVectors(vectorSize, curr, prev);
    }
    cout << "Parallel OpenMP calculate iterations: " << count;
}

void sequentialCalculate(int vectorSize, float* alpha, float* beta, float* prev, float* curr)
{
    int count = 0, criticalCount = CRITICAL_COUNT;
    for (count = 0; count < criticalCount; count++)
    {
        sequentialMultiMatrixVector(vectorSize, alpha, prev, curr);
        sequentialSubVectors(vectorSize, beta, curr, curr);
        if (sequentialIsComplete(vectorSize, prev, curr)) break;
        sequentialCopyVectors(vectorSize, curr, prev);
    }
    cout << "Sequential calculate iterations: " << count;
}

void parallelCudaCalculate(int vectorSize, float* alpha, float* beta, float* prev, float* curr)
{
    float* dev_alpha;
    float* dev_beta;
    float* dev_prev;
    float* dev_curr;
    
    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaMalloc((void**)&dev_alpha, (N * N) * sizeof(float));
    cudaMalloc((void**)&dev_beta, N * sizeof(float));
    cudaMalloc((void**)&dev_prev, N * sizeof(float));
    cudaMalloc((void**)&dev_curr, N * sizeof(float));

    // Copy input matrixes from host memory to GPU buffers.
    cudaMemcpy(dev_alpha, alpha, (N * N) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_beta, beta, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_prev, prev, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_curr, curr, N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch a kernel on the GPU with one thread for each element.
    int blockSize, gridSize;
    // Number of threads in each thread block
    blockSize = 1024;
    // Number of thread blocks in grid
    gridSize = (int)ceil((float)N / blockSize);

    // Global memory
    int count = 0, criticalCount = CRITICAL_COUNT;
    bool *flag = false;
    for (count = 0; count < criticalCount; count++)
    {
        parallelCudaMultiMatrixVectorKernel << <gridSize, blockSize >> > (dev_alpha, dev_prev, dev_curr);
        cudaDeviceSynchronize();
        parallelCudaSubVectorsKernel << <gridSize, blockSize >> > (dev_beta, dev_curr, dev_curr);
        cudaDeviceSynchronize();
        parallelCudaIsComplete << <gridSize, blockSize >> > (dev_prev, dev_curr, flag);
        cudaDeviceSynchronize();
        if (flag) break;
        parallelCudaCopyVectorsKernel << <gridSize, blockSize >> > (dev_curr, dev_prev);
        cudaDeviceSynchronize();
    }

    cout << "Parallel CUDA calculate iterations: " << count;

    // Copy output array from GPU buffer to host memory.
    cudaMemcpy(alpha, dev_alpha, (N * N) * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(beta, dev_beta, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(curr, dev_curr, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(prev, dev_prev, N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dev_alpha);
    cudaFree(dev_beta);
    cudaFree(dev_prev);
    cudaFree(dev_curr);
}

int main()
{
    srand(time(NULL));
    cudaEvent_t start, stop;
    float KernelTime;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

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
    cudaEventRecord(start, 0);

    sequentialCalculate(matrixSize, alpha, beta, prev, curr);

    cudaThreadSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&KernelTime, start, stop);

    cout << " by time: " << round(KernelTime * 100) / 100 << " ms\n";
    // -- >

    initNull(matrixSize, prev);
    initNull(matrixSize, curr);
    
    // <-- Parallel OpenMP method
    cudaEventRecord(start, 0);

    parallelOpenMPCalculate(matrixSize, alpha, beta, prev, curr);

    cudaThreadSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&KernelTime, start, stop);

    cout << " by time: " << round(KernelTime * 100) / 100 << " ms\n";
    // -- >

    initNull(matrixSize, prev);
    initNull(matrixSize, curr);

    // <-- Parallel CUDA method
    cudaEventRecord(start, 0);

    parallelCudaCalculate(matrixSize, alpha, beta, prev, curr);

    cudaThreadSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&KernelTime, start, stop);

    cout << " by time: " << round(KernelTime * 100) / 100 << " ms\n";
    // -- >

    return 0;
}
