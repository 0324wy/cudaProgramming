#include <cublasLt.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cstdint>

int main() {
    cublasLtHandle_t ltHandle;
    cublasLtCreate(&ltHandle);

    int M = 128, N = 128, K = 128;

    int32_t h_A[M * K], h_B[K * N], h_C[M * N];

    for (int i = 0; i < M * K; i++) h_A[i] = (i % 100) + 1;
    for (int i = 0; i < K * N; i++) h_B[i] = 1;

    int32_t *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, M * K * sizeof(int32_t));
    cudaMalloc((void**)&d_B, K * N * sizeof(int32_t));
    cudaMalloc((void**)&d_C, M * N * sizeof(int32_t));

    cudaMemcpy(d_A, h_A, M * K * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(int32_t), cudaMemcpyHostToDevice);

    cublasLtMatmulDesc_t opDesc;
    cublasLtMatmulDescCreate(&opDesc, CUBLAS_COMPUTE_32I, CUDA_R_32I);

    cublasOperation_t transA = CUBLAS_OP_N;
    cublasOperation_t transB = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA));
    cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB));

    cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc;
    cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_32I, M, K, K); // Leading dimension K for A
    cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_32I, K, N, N); // Leading dimension N for B
    cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32I, M, N, N); // Leading dimension N for C

    int32_t alpha = 1;
    int32_t beta = 0;

    cublasStatus_t status = cublasLtMatmul(ltHandle, opDesc, &alpha, d_A, Adesc, d_B, Bdesc, &beta, d_C, Cdesc, d_C, Cdesc, nullptr, nullptr, 0, 0);
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLASLt matrix multiplication failed with error code: " << status << std::endl;
        return -1;
    }

    cudaDeviceSynchronize(); // Ensure all operations are complete before copying back

    cudaMemcpy(h_C, d_C, M * N * sizeof(int32_t), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 100; i++) {
        std::cout << h_C[i] << " ";
    }
    std::cout << std::endl;

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