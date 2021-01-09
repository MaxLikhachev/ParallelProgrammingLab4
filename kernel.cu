#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <omp.h>
#include <cmath>

using namespace std;

#define BLOCK_SIZE 64
#define N 32

#define EPS 0.000001;

void display(int arraySizeDepth, int arraySizeWidth, float* matrix)
{
    for (int i = 0; i < arraySizeDepth; i++)
    {
        for (int j = 0; j < arraySizeWidth; j++)
            cout << matrix[i * arraySizeDepth + j] << " ";
        cout << endl;
    }
}

void display(int arraySize, float* array)
{
    for (int i = 0; i < arraySize; i++)
        cout << array[i] << " ";
    cout << endl;
}

void initRandom(int arraySize, float* array)
{
    for (int i = 0; i < arraySize; i++)
        array[i] = ( 0 + rand() % 100 ) * 0.1;
}

void initRandom(int arraySizeDepth, int arraySizeWidth, float* matrix)
{
    for (int i = 0; i < arraySizeDepth; i++)
        for (int j = 0; j < arraySizeWidth; j++)
        {
            matrix[i * arraySizeDepth + j] = ( 0 + rand() % 100 ) * 0.1;
            if (i == j) matrix[i * arraySizeDepth + j] *= 10;
        }
}

void initNull(int arraySize, float* array)
{
    for (int i = 0; i < arraySize; i++)
        array[i] = 0;
}

void initNull(int arraySizeDepth, int arraySizeWidth, float* matrix)
{
    for (int i = 0; i < arraySizeDepth; i++)
        for (int j = 0; j < arraySizeWidth; j++)
            matrix[i * arraySizeDepth + j] = 0.0;
}

void multi(int arraySize, float* matrix, float* array, float* result)
{
    for (int i = 0; i < arraySize; i++)
        for (int j = 0; j < arraySize; j++)
            result[i] += matrix[i * arraySize + j] * array[j];
}

void init(int arraySize, float* matrix, float* basis, float *temp, float* result)
{
    initNull(arraySize, temp);
    initNull(arraySize, basis);

    initRandom(arraySize, result);
    initRandom(arraySize, arraySize, matrix);

    multi(arraySize, matrix, result, basis);
}


void sequentialCalculate(float* matrix, float* basis, float* result, unsigned int size)
{
    float* temp = new float[size];

    int count;

    for (count = 0; true; count++)
    {
        for (int i = 0; i < size; i++)        
            temp[i] = basis[i];

        for (int i = 0; i < size; i++)
            for (int j = 0; j < size; j++)
                temp[i] -= matrix[i * size + j] * result[j];

        for (int i = 0; i < size; i++)
            temp[i] /= matrix[i * size + i];

        float error = 0.0, eps = EPS;

        for (int i = 0; i < size; i++)
            error += abs(temp[i] - result[i]);

        if (error < eps)
            break;
        
        for (int i = 0; i < size; i++)
            result[i] = temp[i];

    }
    cout << "Sequential calculate count: " << count;
}

int main()
{
    srand(time(NULL));

    cout << "Enter array size: ";
    int arraySize = 0;
    cin >> arraySize;

    float* matrix = new float[arraySize * (arraySize + 1)];
    float* result = new float[arraySize];
    float* basis = new float[arraySize];
    float* temp = new float[arraySize];

    init(arraySize, matrix, basis, temp, result);
    
    cout << "Matrix:\n";
    display(arraySize, arraySize, matrix);
    cout << "Basis:\n";
    display(arraySize, basis);
    cout << "Result:\n";
    display(arraySize, result);
    cout << endl;
    // sequentialCalculate(matrix, basis, result, arraySize);

    return 0;
}
