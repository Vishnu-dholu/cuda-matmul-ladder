// main.cu: benchmark harness.      Usage: ./build/matmul N [reps]
// Prints CSV to stdout; messages go to stderr so scripts can parse stdout cleanly.

#include <cstdio>
#include <cstdlib>
#include <functional>
#include <vector>

#include "common.cuh"
#include "kernels/v1_naive.cuh"
// #include "kernels/v2_coalesced.cuh"
// #include "kernels/v3_tiled.cuh"
// #include "kernels/v4_regtiled.cuh"

// Average time of one launch in milliseconds, measured with CUDA events.
static float time_ms(const std::function<void()>& launch, int reps) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    launch();  // warm-up run
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(start));
    for (int r = 0; r < reps; ++r) launch();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / reps;
}

int main(int argc, char** argv) {
    const int N = (argc > 1) ? atoi(argv[1]) : 1024;
    const int reps = (argc > 2) ? atoi(argv[2]) : 10;

    if (N <= 0 || N % 128 != 0 || reps <= 0) {
        fprintf(stderr, "usage: %s N [reps]  (N must be a positive multiple of 128)\n", argv[0]);
        return 1;
    }

    const size_t count = (size_t)N * N;
    const size_t bytes = count * sizeof(float);

    // 1) random inputs on the host
    std::vector<float> hA(count), hB(count), hRef(count), hOut(count);
    srand(42);
    for (auto& x : hA) x = rand() / (float)RAND_MAX - 0.5f;
    for (auto& x : hB) x = rand() / (float)RAND_MAX - 0.5f;

    // 2) copy to the GPU
    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dB, bytes));
    CUDA_CHECK(cudaMalloc(&dC, bytes));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), bytes, cudaMemcpyHostToDevice));

    // 3) cuBLAS: the reference answer and the speed ceiling
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    const float alpha = 1.0f, beta = 0.0f;
    auto run_cublas = [&] {  // cuBLAS is column-major, so swap A and B to get row-major C = A * B
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, dB, N, dA, N,
                                 &beta, dC, N));
    };
    run_cublas();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hRef.data(), dC, bytes, cudaMemcpyDeviceToHost));

    // 4) for small N, prove the cuBLAS reference against the CPU once
    if (N <= 512) {
        std::vector<float> hCpu(count);
        matmul_cpu(hA.data(), hB.data(), hCpu.data(), N);
        fprintf(stderr, "cuBLAS vs CPU check (N=%d): %s\n", N,
                check(hCpu.data(), hRef.data(), count) ? "OK" : "FAILED");
    }

    // 5) the kernels to benchmark
    dim3 block32(32, 32);
    dim3 grid32((N + 31) / 32, (N + 31) / 32);

    struct Version {
        const char* name;
        std::function<void()> launch;
    };
    std::vector<Version> versions = {
        {"v1_naive", [&] { matmul_naive<<<grid32, block32>>>(dA, dB, dC, N); }},
        {"cublas", run_cublas},
    };

    // 6) verify, time, print one CSV row per kernel
    printf("kernel,N,ms,gflops,correct\n");
    for (auto& v : versions) {
        CUDA_CHECK(cudaMemset(dC, 0, bytes));
        v.launch();
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(hOut.data(), dC, bytes, cudaMemcpyDeviceToHost));
        bool ok = check(hRef.data(), hOut.data(), count);
        float ms = time_ms(v.launch, reps);
        double gflops = 2.0 * N * N * N / (ms * 1e-3) / 1e9;
        printf("%s,%d,%.4f,%.1f,%d\n", v.name, N, ms, gflops, ok ? 1 : 0);
        fflush(stdout);
    }

    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    return 0;
}
