#include <iostream>
#include <cuda_runtime.h>
using namespace std;

__global__ void vecAddition(int *d_a, int *d_b, int *d_c, int n)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;

    if (i < n)
    {
        d_c[i] = d_a[i] + d_b[i];
    }
}

void printArray(int *h_num, int n)
{
    for (size_t i = 0; i < n; i++)
    {
        cout << h_num[i] << endl;
    }
}

int main()
{
    // create two array
    int n = 10;
    int size = sizeof(int) * n;

    int *h_a = (int *)malloc(size);
    int *h_b = (int *)malloc(size);
    int *h_c = (int *)malloc(size);

    for (size_t i = 0; i < n; i++)
    {
        h_a[i] = static_cast<int>(i);
        h_b[i] = static_cast<int>(i * 2);

    }

    // printArray(h_a, n);
    // printArray(h_b, n);

    // allocate cuda memory
    int *d_a;
    int *d_b;
    int *d_c;

    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    vecAddition<<<ceil(n/256.0), 256>>>(d_a, d_b, d_c, n);

    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    printArray(h_c, n);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}

