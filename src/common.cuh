// common.cuh: error-checking macros, CPU reference and correctness checker
#pragma once
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>

// Stop with a clear message if a CUDA call fails.
#define CUDA_CHECK(call)                                                                     \
    do {                                                                                     \
        cudaError_t err_ = (call);                                                           \
        if (err_ != cudaSuccess) {                                                           \
            fprintf(stderr, "CUDA error: %s at %s:%d\n", cudaGetErrorString(err_), __FILE__, \
                    __LINE__);                                                               \
            exit(EXIT_FAILURE);                                                              \
        }                                                                                    \
    } while (0)

// Same idea for cuBLAS calls.
#define CUBLAS_CHECK(call)                                                               \
    do {                                                                                 \
        cublasStatus_t st_ = (call);                                                     \
        if (st_ != CUBLAS_STATUS_SUCCESS) {                                              \
            fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)st_, __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                                          \
        }                                                                                \
    } while (0)

// Slow but simple CPU matmul, used once to prove the cuBLAS reference is right.
inline void matmul_cpu(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < N; ++k) acc += A[i * N + k] * B[k * N + j];
            C[i * N + j] = acc;
        }
}

// Compare two results with a relative tolerance (never use == on floats).
inline bool check(const float* ref, const float* got, size_t n, float tol = 1e-3f) {
    for (size_t i = 0; i < n; ++i) {
        float denom = fmaxf(1.0f, fabsf(ref[i]));
        if (fabsf(ref[i] - got[i]) / denom > tol) {
            fprintf(stderr, "  mismatch at %zu: ref %f, got %f\n", i, ref[i], got[i]);
            return false;
        }
    }
    return true;
}
