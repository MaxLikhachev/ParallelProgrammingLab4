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
    cout << "Finished with ierations count: " << count;
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
    cout << "Finished with iterations count: " << count;
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

    cudaEvent_t start, stop;
    float KernelTime;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    sequentialCalculate(matrixSize, alpha, beta, prev, curr);
    cudaThreadSynchronize();

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&KernelTime, start, stop);
    cout << "\nSequential calculate time: " << KernelTime * 1000 << " mcs\n";

    // -- >

    initNull(matrixSize, prev);
    initNull(matrixSize, curr);
    cudaEventRecord(start, 0);

    // <-- Parallel OpenMP method

    parallelOpenMPCalculate(matrixSize, alpha, beta, prev, curr);
    cudaThreadSynchronize();

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&KernelTime, start, stop);
    cout << "\nParallel OpenMP calculate time: " << KernelTime * 1000 << " mcs\n";

    // -- >

    initNull(matrixSize, prev);
    initNull(matrixSize, curr);

    // <-- Parallel CUDA method

    cudaEventRecord(start, 0);

    parallelOpenMPCalculate(matrixSize, alpha, beta, prev, curr);
    cudaThreadSynchronize();

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&KernelTime, start, stop);
    cout << "\nParallel OpenMP calculate time: " << KernelTime * 1000 << " mcs\n";


    // -- >

    return 0;
}
