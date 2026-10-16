// v1_naive.cuh: one thread computes one element of C.
// threadIdx.x selects the ROW on purpose. This is uncoalesced; v2 fixes it.

#pragma once

__global__ void matmul_naive(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < N && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < N; k++) {
            acc += A[row * N + k] * B[k * N + col];
        }

        C[row * N + col] = acc;
    }
}
