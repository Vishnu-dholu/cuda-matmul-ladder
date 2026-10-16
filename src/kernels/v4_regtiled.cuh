// v4_regtiled.cuh: register tiling (thread coarsening).
// A block computes a BM x BN tile of C; each thread computes a TM x TN patch kept in registers.
// Launch: dim3 block((BM*BN)/(TM*TN)) (1D), dim3 grid(N/BN, N/BM).
// Assumes N is a multiple of BM, BN and BK (the harness requires N % 128 == 0).
#pragma once

template <int BM, int BN, int BK, int TM, int TN>
__global__ void matmul_regtiled(const float* A, const float* B, float* C, int N) {
    constexpr int NT = (BM * BN) / (TM * TN);  // threads per block
    static_assert(NT % BK == 0 && NT % BN == 0, "thread count must divide the tile loads evenly");
    static_assert((BM * BK) % NT == 0 && (BK * BN) % NT == 0,
                  "tile loads must split evenly across threads");

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    const int threadCol = threadIdx.x % (BN / TN);  // which group of TN columns I own
    const int threadRow = threadIdx.x / (BN / TN);  // which group of TM rows I own

    // move the pointers to this block's tile
    A += blockIdx.y * BM * N;  // first row of A for this block
    B += blockIdx.x * BN;      // first column of B for this block
    C += blockIdx.y * BM * N + blockIdx.x * BN;

    // how this thread helps fill the shared tiles (neighbours read neighbours: coalesced)
    const int aRow = threadIdx.x / BK, aCol = threadIdx.x % BK, aStride = NT / BK;
    const int bRow = threadIdx.x / BN, bCol = threadIdx.x % BN, bStride = NT / BN;

    float acc[TM * TN] = {0.0f};  // my results live in registers
    float regA[TM], regB[TN];

    for (int bk = 0; bk < N; bk += BK) {
        // 1) cooperative load of a BM x BK slice of A and a BK x BN slice of B
        for (int r = 0; r < BM; r += aStride) As[(aRow + r) * BK + aCol] = A[(aRow + r) * N + aCol];
        for (int r = 0; r < BK; r += bStride) Bs[(bRow + r) * BN + bCol] = B[(bRow + r) * N + bCol];
        __syncthreads();
        A += BK;  // slide along K
        B += BK * N;

        // 2) for each k: TM values of A and TN values of B into registers, then TM*TN FMAs
        for (int k = 0; k < BK; ++k) {
            for (int i = 0; i < TM; ++i) regA[i] = As[(threadRow * TM + i) * BK + k];
            for (int j = 0; j < TN; ++j) regB[j] = Bs[k * BN + threadCol * TN + j];
            for (int i = 0; i < TM; ++i)
                for (int j = 0; j < TN; ++j) acc[i * TN + j] += regA[i] * regB[j];
        }
        __syncthreads();
    }

    // 3) write my TM x TN results
    for (int i = 0; i < TM; ++i)
        for (int j = 0; j < TN; ++j)
            C[(threadRow * TM + i) * N + threadCol * TN + j] = acc[i * TN + j];
}
