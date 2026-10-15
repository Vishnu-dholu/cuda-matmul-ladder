#include <cuda_runtime.h>

#include <cstdio>

static int attr(cudaDeviceAttr a) {
    int v = 0;
    cudaDeviceGetAttribute(&v, a, 0);
    return v;
}

int main() {
    cudaDeviceProp p;
    if (cudaGetDeviceProperties(&p, 0) != cudaSuccess) {
        printf("Nod CUDA device found\n");
        return 1;
    }

    int sms = attr(cudaDevAttrMultiProcessorCount);
    int clockKHz = attr(cudaDevAttrClockRate);
    int memKHz = attr(cudaDevAttrMemoryClockRate);
    int busBits = attr(cudaDevAttrGlobalMemoryBusWidth);
    const int coresPerSM = 64;

    double peakGflops = 2.0 * sms * coresPerSM * clockKHz / 1e6;
    double peakGBs = 2.0 * memKHz * 1.e3 * (busBits / 8.0) / 1e9;

    printf("GPU: %s (compute capability %d.%d)\n", p.name, p.major, p.minor);
    printf("SMs: %d | FP32 cores: %d | max clock: %.0f MHz\n", sms, sms * coresPerSM,
           clockKHz / 1e3);
    printf("Memory: %.1f GB | bus: %d-bit | L2: %d KB\n", p.totalGlobalMem / 1e9, busBits,
           attr(cudaDevAttrL2CacheSize) / 1024);
    printf("Shared mem per block: %d KB | registers per SM: %d | max threads per SM: %d\n",
           attr(cudaDevAttrMaxSharedMemoryPerBlock) / 1024,
           attr(cudaDevAttrMaxRegistersPerMultiprocessor),
           attr(cudaDevAttrMaxThreadsPerMultiProcessor));
    printf(
        "Theoretical FP32 peak: %.0f GFLOPS | bandwidth: %.0f GB/s | ridge point: %.1f FLOP/byte\n",
        peakGflops, peakGBs, peakGflops / peakGBs);
    printf("Kernel time limit (display watchdog): %s\n",
           attr(cudaDevAttrKernelExecTimeout) ? "yes" : "no");
    return 0;
}
