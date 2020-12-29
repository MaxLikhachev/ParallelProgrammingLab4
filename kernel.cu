#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <omp.h>

using namespace std;

#define BLOCK_SIZE 64
#define N 32

const float eps = 0.000001;

cudaError_t calculateWithCuda(float* matrix, unsigned int size);

__global__ void globalCalculateKernel(float* c, float* a, float* b)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    c[i * j] = sin(a[i * j]) * sin(a[i * j]) + cos(b[i * j]) * cos(b[i * j]) * cos(b[i * j]);
}

__global__ void sumKernel(float* matrix, float* sum, float* result, float* temp)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    // if (i != j)
        // sum += matrix[i * j] * result[j];

    // temp[i] = (matrix[i * size + size] - sum) / matrix[i * size + i];
    // c[i * j] = sin(a[i * j]) * sin(a[i * j]) + cos(b[i * j]) * cos(b[i * j]) * cos(b[i * j]);
}

bool isDiverged(float* result, float* temp, unsigned int size)
{
    bool flag = true;
    for (int i = 0; i < size && flag; i++)
        flag = fabs(temp[i] - result[i]) < eps;
    return flag;
}

bool isDiagonalDominanceBroken(float *matrix, unsigned int size)
{
    bool flag = true;
    for (int i = 0; i < size && flag; i++)
    {
        float fabsSum = 0.0;
        for (int j = 0; j < size; j++)
            if (i != j)
                fabsSum += fabs(matrix[i * size + j]);
        flag = fabs(matrix[i * size + i]) <= fabsSum;
    }
    return flag;
}


void initRandom(int arraySize, float* a)
{
    for (int i = 0; i < arraySize; i++)
        for (int j = 0; j < arraySize; j++)
        {
            a[i * arraySize + j] = 0 + rand() % arraySize;
            if (i == j) a[i * arraySize + j] += arraySize * arraySize;
        }       
}

void initMatrixNull(int arraySize, float* a)
{
    for (int i = 0; i < arraySize; i++)
        for (int j = 0; j < arraySize; j++)
            a[i * arraySize + j] = 0;
}

void initNull(int arraySize, float* a)
{
    for (int i = 0; i < arraySize; i++)
        a[i] = 0;
}

void display(int arraySize, float* a)
{
    for (int i = 0; i < arraySize; i++)
    {
        for (int j = 0; j < arraySize; j++)
            cout << a[i * arraySize + j] << " ";
        cout << endl;
    }
}

float* parallelOpenMPCalculate(float* matrix, unsigned int size)
{
    float* result = new float[size];
    float* temp = new float[size];
    initNull(size, result);
    initNull(size, temp);

    int count = 0;
    for (bool flag = !isDiagonalDominanceBroken(matrix, size); flag; count++)
    {
#pragma omp parallel for
        for (int i = 0; i < size; i++)
        {
            float sum = 0.0;
            {
#pragma omp parallel for
                for (int j = 0; j < size; j++)
                    if (i != j)
                        sum += matrix[i * size + j] * result[j];
                temp[i] = (matrix[i * size + size] - sum) / matrix[i * size + i];
            }
        }
        flag = !isDiverged(result, temp, size);
        if (flag)
            for (int i = 0; i < size; i++)
                result[i] = temp[i];
    }
    cout << "Parallel OpenMP count: " << count;
    return result;
}

float* sequentialCalculate(float* matrix, unsigned int size)
{
    float* result = new float[size];
    float* temp = new float[size];
    initNull(size, result);
    initNull(size, temp);
    
    int count = 0;
    for (bool flag = !isDiagonalDominanceBroken(matrix, size); flag; count++)
    {
        for (int i = 0; i < size; i++)
        {
            float sum = 0.0;
            for (int j = 0; j < size; j++)
                if (i != j)
                    sum += matrix[i * size + j] * result[j];
            temp[i] = (matrix[i * size + size] - sum) / matrix[i * size + i];
        }
        flag = !isDiverged(result, temp, size);
        if (flag)
            for (int i = 0; i < size; i++)
                result[i] = temp[i];
    }
    cout << "Sequential calculate count: " << count;
    return result;
}

int main()
{
    srand(time(NULL));

    cout << "Enter array size: ";
    int arraySize = 0;
    cin >> arraySize;

    float* matrix = new float[arraySize * arraySize];

    initRandom(arraySize, matrix);
    // display(arraySize, matrix);

    cudaEvent_t start, stop;
    float KernelTime;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    sequentialCalculate(matrix, arraySize);

    cudaThreadSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&KernelTime, start, stop);
    printf("\nSequential calculate time:  %0.2f ms \n", KernelTime);

    cudaEventRecord(start, 0);

    parallelOpenMPCalculate(matrix, arraySize);

    cudaThreadSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&KernelTime, start, stop);
    printf("\nParallel OpenMP calculate time:  %0.2f ms \n", KernelTime);

    // Add matrixes in parallel.
    /*
    cudaError_t cudaStatus = calculateWithCuda(c, a, b, arraySize);
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
    */
    return 0;
}


// Helper function for using CUDA to add matrixes in parallel.
cudaError_t calculateWithCuda(float* matrix, unsigned int size)
{
    float* dev_matrix;

    cudaError_t cudaStatus;
    cudaEvent_t start, stop;
    float KernelTime;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for matrix.
    cudaEventRecord(start, 0);
    cudaStatus = cudaMalloc((void**)&dev_matrix, (N * N) * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaThreadSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&KernelTime, start, stop);
    printf("\nAllocating GPU buffers time:  %0.2f ms \n", KernelTime);

    // Copy input matrixes from host memory to GPU buffers.
    cudaEventRecord(start, 0);

    cudaStatus = cudaMemcpy(dev_matrix, matrix, (N * N) * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaThreadSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&KernelTime, start, stop);
    printf("\nCopying matrix: host -> GPU  time:  %0.2f ms \n", KernelTime);

    // Launch a kernel on the GPU with one thread for each element.
    int numBlocks = BLOCK_SIZE;
    dim3 threadsPerBlock(N, N);
    cout << "\nConfig settings: arraySize = " << size << ", numBlocks = " << numBlocks << ", threadsPerBlock(" << N << ", " << N << ")\n";

    // Global memory
    cudaEventRecord(start, 0);
    // globalCalculateKernel << <numBlocks, threadsPerBlock >> > (dev_c, dev_a, dev_b);
    cudaThreadSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&KernelTime, start, stop);
    // cout << "\nGlobal result: " << KernelTime <<  " milliseconds\n";
    printf("\nGlobal memory work's time:  %0.2f ms \n", KernelTime);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "globalCalculateKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching globalCalculateKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output matrix from GPU buffer to host memory.
    cudaEventRecord(start, 0);
    cudaStatus = cudaMemcpy(matrix, dev_matrix, (N * N) * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    cudaThreadSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&KernelTime, start, stop);
    printf("\nCopying output matri: GPU -> host time:  %0.2f ms \n", KernelTime);

Error:
    cudaFree(dev_matrix);

    return cudaStatus;
}