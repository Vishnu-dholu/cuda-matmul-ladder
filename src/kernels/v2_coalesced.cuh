// V2_coalesced.cuh: same math as v1, but threadIdx.x now selects the COLUMN,
// so the 32 threads of a warp touch neighbouring addresses.

#pragma once

__global__ void matmul_coalesced(const float* A, const float* B, float* C, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < N && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < N; k++) {
            acc += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = acc;
    }
}
