// v3_tiled.cuh: shared-memory tiling. A bloack computes a TILE x TILE tile of C.
// Launch with dim3 block(TILE, TILE) and grid(ceil(N/TILE), ceil(N/TILE)).

#pragma once

template <int TILE>
__global__ void matmul_tiled(const float* A, const float* B, float* C, int N) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int tx = threadIdx.x, ty = threadIdx.y;
    int col = blockIdx.x * TILE + tx;
    int row = blockIdx.y * TILE + ty;

    float acc = 0.0f;
    int numTiles = (N + TILE - 1) / TILE;

    for (int t = 0; t < numTiles; t++) {
        // 1) coorperative, coalesced load: one element of each tile per thread
        int aCol = t * TILE + tx;
        int bRow = t * TILE + ty;
        As[ty][tx] = (row < N && aCol < N) ? A[row * N + aCol] : 0.0f;
        Bs[ty][tx] = (bRow < N && col < N) ? B[bRow * N + col] : 0.0f;
        __syncthreads();  // the while tile has arrived

        // 2) compute from fast shared memory
        for (int k = 0; k < TILE; k++) {
            acc += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();  // everyone is done before the next load overwrites
    }

    if (row < N && col < N) C[row * N + col] = acc;
}
