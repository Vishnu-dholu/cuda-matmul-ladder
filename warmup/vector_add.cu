// vector_add.cu      build: nvcc vector_add.cu -o vector_add
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>

// Wrap every CUDA call: if it fails, print where and why, then exit.
#define CUDA_CHECK(call)                                                                    \
    do {                                                                                    \
        cudaError_t err = (call);                                                           \
        if (err != cudaSuccess) {                                                           \
            fprintf(stderr, "CUDA error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, \
                    __LINE__);                                                              \
            exit(EXIT_FAILURE);                                                             \
        }                                                                                   \
    } while (0)

// The kernel runs on the GPU, one thread per element.
__global__ void vecAdd(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // which element do I own?
    if (i < n) {                                    // spare threads in the last block do nothing
        c[i] = a[i] + b[i];
    }
}

int main() {
    const int n = 1 << 20;  // 1,048,576 elements
    const size_t bytes = n * sizeof(float);

    // 1) host (CPU) arrays: prefix h_
    float* h_a = (float*)malloc(bytes);
    float* h_b = (float*)malloc(bytes);
    float* h_c = (float*)malloc(bytes);
    for (int i = 0; i < n; ++i) {
        h_a[i] = 1.0f * i;
        h_b[i] = 2.0f * i;
    }

    // 2) device (GPU) arrays: prefix d_
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    // 3) copy inputs CPU -> GPU
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    // 4) launch with enough blocks to cover n elements
    int threadsPerBlock = 256;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    vecAdd<<<blocks, threadsPerBlock>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaGetLastError());       // catches launch errors (bad configuration)
    CUDA_CHECK(cudaDeviceSynchronize());  // waits, and catches errors during execution

    // 5) copy the result GPU -> CPU and verify
    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; ++i) {
        if (fabsf(h_c[i] - 3.0f * i) > 1e-3f) {
            printf("Mismatch at %d\n", i);
            return 1;
        }
    }
    printf("vector add OK\n");

    // 6) clean up
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);
    return 0;
}
