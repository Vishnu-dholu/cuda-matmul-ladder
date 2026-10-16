// cache_effects.cpp
// Build: g++ -O2 -std=c++17 cache_effects.cpp -o cache_effects
// Use -O2 (not -O3): -O3 may interchange the loops and hide the effect.
#include <chrono>
#include <cstdio>
#include <vector>

using Clock = std::chrono::steady_clock;
static double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

int main() {
    // ---- Experiment 1: row-major vs column-major traversal ----
    const int N = 4096;  // 4096 x 4096 ints = 64 MB
    std::vector<int> m(static_cast<size_t>(N) * N, 1);

    auto t0 = Clock::now();
    long long sum_row = 0;
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            sum_row += m[static_cast<size_t>(i) * N + j];  // next element is a neighbour
    double t_row = ms_since(t0);

    t0 = Clock::now();
    long long sum_col = 0;
    for (int j = 0; j < N; ++j)
        for (int i = 0; i < N; ++i)
            sum_col += m[static_cast<size_t>(i) * N + j];  // jumps N*4 bytes every step
    double t_col = ms_since(t0);

    std::printf("row-major %.1f ms | column-major %.1f ms | %.1fx slower (sums %lld %lld)\n", t_row,
                t_col, t_col / t_row, sum_row, sum_col);

    // ---- Experiment 2: same work, different stride ----
    const size_t SIZE = size_t(1) << 25;  // 32M ints = 128 MB
    std::vector<int> a(SIZE, 1);
    for (size_t stride = 1; stride <= 1024; stride *= 2) {
        long long s = 0;
        t0 = Clock::now();
        for (size_t offset = 0; offset < stride; ++offset)  // every element is touched exactly once
            for (size_t i = offset; i < SIZE; i += stride) s += a[i];
        std::printf("stride %5zu ints (%6zu bytes): %.2f ns/element (sum %lld)\n", stride,
                    stride * sizeof(int), ms_since(t0) * 1e6 / SIZE, s);
    }
}
