// main.cu: benchmark harness.
// Usage: ./build/matmul N [reps] [--csv]
//   default : readable table in the terminal
//   --csv   : machine-readable CSV on stdout (used by scripts/run_bench.py)
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>

#include "common.cuh"
#include "kernels/v1_naive.cuh"
#include "kernels/v2_coalesced.cuh"
#include "kernels/v3_tiled.cuh"
// Step 6: #include "kernels/v4_regtiled.cuh"

struct Result {
    std::string name;
    float ms;
    double gflops;
    bool ok;
};

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

static void print_csv(const std::vector<Result>& res, int N) {
    printf("kernel,N,ms,gflops,correct\n");
    for (const auto& r : res)
        printf("%s,%d,%.4f,%.1f,%d\n", r.name.c_str(), N, r.ms, r.gflops, r.ok ? 1 : 0);
}

static void print_table(const std::vector<Result>& res, int N, int reps, const char* gpu,
                        bool color) {
    const char* bold = color ? "\033[1m" : "";
    const char* green = color ? "\033[32m" : "";
    const char* red = color ? "\033[31m" : "";
    const char* reset = color ? "\033[0m" : "";

    double base = res.front().gflops;  // first row = v1_naive
    double ceiling = 0.0;              // cuBLAS result
    for (const auto& r : res)
        if (r.name == "cublas") ceiling = r.gflops;

    printf("\n%sGPU:%s %s   %sN:%s %d   %sreps:%s %d\n\n", bold, reset, gpu, bold, reset, N, bold,
           reset, reps);
    printf("%s  %-14s %11s %10s %9s %10s  %-5s  %s%s\n", bold, "Kernel", "Time (ms)", "GFLOPS",
           "vs v1", "% cuBLAS", "Check", "Share of cuBLAS", reset);
    printf("  %s\n", std::string(86, '-').c_str());
    for (const auto& r : res) {
        double pct = ceiling > 0.0 ? 100.0 * r.gflops / ceiling : 0.0;
        int bar = (int)(pct / 5.0 + 0.5);  // 20 characters = 100%
        if (bar > 20) bar = 20;
        printf("  %-14s %11.3f %10.1f %8.1fx %9.1f%%  %s%-5s%s  %s\n", r.name.c_str(), r.ms,
               r.gflops, r.gflops / base, pct, r.ok ? green : red, r.ok ? "PASS" : "FAIL", reset,
               std::string(bar, '#').c_str());
    }
    printf("\n");
}

int main(int argc, char** argv) {
    // ---- arguments: two optional numbers (N, reps) and an optional --csv flag ----
    bool csv = false;
    std::vector<int> nums;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--csv")
            csv = true;
        else
            nums.push_back(atoi(argv[i]));
    }
    const int N = nums.size() > 0 ? nums[0] : 1024;
    const int reps = nums.size() > 1 ? nums[1] : 10;
    if (N <= 0 || N % 128 != 0 || reps <= 0) {
        fprintf(stderr, "usage: %s N [reps] [--csv]   (N must be a positive multiple of 128)\n",
                argv[0]);
        return 1;
    }
    const bool color = !csv && isatty(STDOUT_FILENO);  // colors only in a real terminal
    const bool progress = !csv && isatty(STDERR_FILENO);

    const size_t count = (size_t)N * N;
    const size_t bytes = count * sizeof(float);

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

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

    // 5) the kernels to benchmark (v1_naive must stay first: it is the "vs v1" baseline)
    dim3 block32(32, 32);
    dim3 grid32((N + 31) / 32, (N + 31) / 32);
    dim3 block16(16, 16);
    dim3 grid16((N + 15) / 16, (N + 15) / 16);

    struct Version {
        const char* name;
        std::function<void()> launch;
    };
    std::vector<Version> versions = {
        {"v1_naive", [&] { matmul_naive<<<grid32, block32>>>(dA, dB, dC, N); }},
        {"v2_coalesced", [&] { matmul_coalesced<<<grid32, block32>>>(dA, dB, dC, N); }},
        {"v3_tiled_t16", [&] { matmul_tiled<16><<<grid16, block16>>>(dA, dB, dC, N); }},
        {"v3_tiled_t32", [&] { matmul_tiled<32><<<grid32, block32>>>(dA, dB, dC, N); }},
        // Step 6: add the v4_regtiled line here
        {"cublas", run_cublas},
    };

    // 6) verify and time every kernel, collect the results
    std::vector<Result> results;
    for (auto& v : versions) {
        if (progress) {
            fprintf(stderr, "\r  running %-16s", v.name);
            fflush(stderr);
        }
        CUDA_CHECK(cudaMemset(dC, 0, bytes));  // so a kernel that does nothing cannot pass
        v.launch();
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(hOut.data(), dC, bytes, cudaMemcpyDeviceToHost));
        bool ok = check(hRef.data(), hOut.data(), count);
        float ms = time_ms(v.launch, reps);
        double gflops = 2.0 * N * N * N / (ms * 1e-3) / 1e9;
        results.push_back({v.name, ms, gflops, ok});
    }
    if (progress) {
        fprintf(stderr, "\r%40s\r", "");
        fflush(stderr);
    }

    // 7) report
    if (csv)
        print_csv(results, N);
    else
        print_table(results, N, reps, prop.name, color);

    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    return 0;
}
