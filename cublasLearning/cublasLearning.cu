#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>

int main() {
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Define matrix dimensions
    int m = 2, n = 2, k = 2;

    // Host matrices
    float h_A[m * k] = {1, 2, 3, 4};
    float h_B[k * n] = {5, 6, 7, 8};
    float h_C[m * n];

    // Device matrices
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, m * k * sizeof(float));
    cudaMalloc((void**)&d_B, k * n * sizeof(float));
    cudaMalloc((void**)&d_C, m * n * sizeof(float));

    // Transfer data to device
    cudaMemcpy(d_A, h_A, m * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, k * n * sizeof(float), cudaMemcpyHostToDevice);

    // Perform matrix multiplication: h_C = h_A * h_B
    const float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_A, m, d_B, k, &beta, d_C, m);

    // Copy result back to host
    cudaMemcpy(h_C, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost);

    // Print result
    printf("Result matrix C:\n");
    for (int i = 0; i < m * n; ++i) printf("%f ", h_C[i]);
    printf("\n");

    // Clean up
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cublasDestroy(handle);

    return 0;
}