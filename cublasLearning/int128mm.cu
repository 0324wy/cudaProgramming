#include <cublasLt.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cstdint>

// Helper function to print __uint128_t as two 64-bit integers
void print_uint128(__uint128_t value) {
    uint64_t high = value >> 64;
    uint64_t low = static_cast<uint64_t>(value);
    std::cout << "High: " << high << ", Low: " << low << std::endl;
}

int main() {
    // Initialize cuBLASLt
    cublasLtHandle_t ltHandle;
    cublasLtCreate(&ltHandle);

    // Define matrix dimensions (example: 128x128)
    int M = 128, N = 128, K = 128;

    // Host matrices (int32)
    __uint128_t h_A[M * K], h_B[K * N], h_C[M * N];

    // Initialize matrices with some values (fill h_A and h_B as needed)
    for (int i = 0; i < M * K; i++) h_A[i] = i;    // Populate h_A
    for (int i = 0; i < K * N; i++) h_B[i] = i * 2; // Populate h_B


    // Device matrices
    __uint128_t *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, M * K * sizeof(__uint128_t));
    cudaMalloc((void**)&d_B, K * N * sizeof(__uint128_t));
    cudaMalloc((void**)&d_C, M * N * sizeof(__uint128_t));

    // Copy data to device
    cudaMemcpy(d_A, h_A, M * K * sizeof(__uint128_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(__uint128_t), cudaMemcpyHostToDevice);

    // Create operation descriptor
    cublasLtMatmulDesc_t opDesc;
    cublasLtMatmulDescCreate(&opDesc, CUBLAS_COMPUTE_32I, CUDA_R_32I);  // Corrected: cublasComputeType_t and cudaDataType_t

    // Set matrix operation attributes
    cublasOperation_t transA = CUBLAS_OP_N;
    cublasOperation_t transB = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA));
    cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB));

    // Create matrix layout descriptors
    cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc;
    cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_32I, M, K, M);  // CUDA_R_32I for int32 data type
    cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_32I, K, N, K);
    cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32I, M, N, M);

    // Scaling factors
    __uint128_t alpha = 1;
    __uint128_t beta = 0;

    // Launch integer GEMM
    cublasLtMatmul(ltHandle, opDesc, &alpha, d_A, Adesc, d_B, Bdesc, &beta, d_C, Cdesc, d_C, Cdesc, nullptr, nullptr, 0, 0);

    // Copy result back to host
    cudaMemcpy(h_C, d_C, M * N * sizeof(__uint128_t), cudaMemcpyDeviceToHost);

    // Print result (first few elements) using the helper function
    for (int i = 0; i < 10; i++) {
        print_uint128(h_C[i]);
    }

    // Cleanup
    cublasLtMatrixLayoutDestroy(Adesc);
    cublasLtMatrixLayoutDestroy(Bdesc);
    cublasLtMatrixLayoutDestroy(Cdesc);
    cublasLtMatmulDescDestroy(opDesc);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasLtDestroy(ltHandle);

    return 0;
}