#include <iostream>
#include <cuda_runtime.h>

using namespace std;

__global__ void matrixMul(float *m, float *n, float *p, int width)
{

    int col = threadIdx.x + blockDim.x * blockIdx.x;
    int row = threadIdx.y + blockDim.y * blockIdx.y;

    if (row < width && col < width)
    {
        float Pvalue = 0;
        for (size_t k = 0; k < width; k++)
        {
            Pvalue += m[row * width + k] * n[k * width + col];
        }
        p[row * width + col] = Pvalue;
    }
}

void printArray(float *h_num, int n)
{
    for (size_t j = 0; j < n; j++)
    {
        for (size_t i = 0; i < n; i++)
        {
            cout << h_num[j * n + i] << endl;
        }
    }
}

int main()
{

    int width = 3;
    int matrixSize = width * width * sizeof(float);

    float *h_m = (float *)malloc(matrixSize);
    float *h_n = (float *)malloc(matrixSize);
    float *h_p = (float *)malloc(matrixSize);

    for (size_t i = 0; i < width; i++)
    {
        for (size_t j = 0; j < width; j++)
        {
            h_m[i * width + j] = static_cast<float>(i);
            h_n[i * width + j] = static_cast<float>(j);
        }
    }

    printArray(h_m, width);

    float *d_m;
    float *d_n;
    float *d_p;
    
    cudaMalloc((void **)&d_m, matrixSize);
    cudaMalloc((void **)&d_n, matrixSize);
    cudaMalloc((void **)&d_p, matrixSize);

    cudaMemcpy(d_m, h_m, matrixSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_n, h_n, matrixSize, cudaMemcpyHostToDevice);

    dim3 threadPerBlock(32, 32);
    dim3 blockPerGrid(ceil(width/static_cast<float>(threadPerBlock.x)), ceil(width/static_cast<float>(threadPerBlock.y)));
    matrixMul<<<blockPerGrid, threadPerBlock>>>(d_m, d_n, d_p, width);

    cudaMemcpy(h_p, d_p, matrixSize, cudaMemcpyDeviceToHost);

    printArray(h_p, width);

    cudaFree(d_m);
    cudaFree(d_n);
    cudaFree(d_p);

    free(h_m);
    free(h_n);
    free(h_p);
}