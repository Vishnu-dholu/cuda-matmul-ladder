#include <cstdio>

__global__ void hello() { printf("Hello from block %d, thread %d\n", blockIdx.x, threadIdx.x); }

int main() {
    hello<<<2, 4>>>();
    cudaDeviceSynchronize();  // wait for the GPU before exiting
    return 0;
}
