#include <iostream>
#include <cuda_runtime.h>

using namespace std;

__global__ void matrixVec(float *inVec, float *inMatrix, float *outVec, int width)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;

    if (i < width)
    {
        float outValue = 0;
        for (size_t j = 0; j < width; j++)
        {
            outValue += inVec[j] * inMatrix[i + j * width];
        }
        outVec[i] = outValue;
    }
}

void printArray(float *h_num, int n)
{
    for (size_t i = 0; i < n; i++)
    {
        cout << h_num[i] << endl;
    }
}

int main()
{
    int width = 3;
    float *h_inVec = (float *)malloc(width * sizeof(float));
    float *h_inMatrix = (float *)malloc(width * width * sizeof(float));
    float *h_outVec = (float *)malloc(width * sizeof(float));

    for (size_t i = 0; i < width; i++)
    {
        h_inVec[i] = static_cast<float>(i);
        for (size_t j = 0; j < width; j++)
        {
            h_inMatrix[i * width + j] = static_cast<float>(j);
        }
    }
    
    float *d_inVec;
    float *d_inMatrix;
    float *d_outVec;

    cudaMalloc((void **)&d_inVec, width * sizeof(float));
    cudaMalloc((void **)&d_inMatrix, width * width * sizeof(float));
    cudaMalloc((void **)&d_outVec, width * sizeof(float));

    cudaMemcpy(d_inVec, h_inVec, width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_inMatrix, h_inMatrix, width * width * sizeof(float), cudaMemcpyHostToDevice);

    matrixVec<<<ceil(width / 64.0), 64>>>(d_inVec, d_inMatrix, d_outVec, width);


    cudaMemcpy(h_outVec, d_outVec, width * sizeof(float), cudaMemcpyDeviceToHost);

    printArray(h_outVec, width);

    cudaFree(d_inVec);
    cudaFree(d_inMatrix);
    cudaFree(d_outVec);

    free(h_inVec);
    free(h_inMatrix);
    free(h_outVec);

    return 0;
}