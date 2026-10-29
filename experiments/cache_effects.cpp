// cache_effects.cpp
// Build: g++ -O2 -std=c++17 cache_effects.cpp -o cache_effects
// Use -O2 (not -O3): -O3 may interchange the loops and hide the effect.
// Every measurement gets one untimed warm-up pass, then REPEATS timed runs; min and median are shown.
// Row and column runs alternate so clock drift or background load hits both equally.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <vector>

using Clock = std::chrono::steady_clock;
static double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

constexpr int REPEATS = 7;

struct Stats {
    double min, median;
};
static Stats stats(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return {v.front(), v[v.size() / 2]};
}

int main() {
    // ---- Experiment 1: row-major vs column-major traversal ----
    const int N = 4096;  // 4096 x 4096 ints = 64 MB
    std::vector<int> m(static_cast<size_t>(N) * N, 1);

    auto row_pass = [&] {
        long long s = 0;
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                s += m[static_cast<size_t>(i) * N + j];  // next element is a neighbour
        return s;
    };
    auto col_pass = [&] {
        long long s = 0;
        for (int j = 0; j < N; ++j)
            for (int i = 0; i < N; ++i)
                s += m[static_cast<size_t>(i) * N + j];  // jumps N*4 bytes every step
        return s;
    };

    long long sink = row_pass() + col_pass();  // warm-up (untimed)
    std::vector<double> row_ms, col_ms;
    for (int r = 0; r < REPEATS; ++r) {
        auto t0 = Clock::now();
        sink += row_pass();
        row_ms.push_back(ms_since(t0));
        t0 = Clock::now();
        sink += col_pass();
        col_ms.push_back(ms_since(t0));
    }
    Stats sr = stats(row_ms), sc = stats(col_ms);
    std::printf("row-major    min %.1f ms, median %.1f ms\n", sr.min, sr.median);
    std::printf("column-major min %.1f ms, median %.1f ms\n", sc.min, sc.median);
    std::printf("column is %.1fx slower (min) / %.1fx (median)  [%d runs, checksum %lld]\n\n",
                sc.min / sr.min, sc.median / sr.median, REPEATS, sink);

    // ---- Experiment 2: same work, different stride ----
    const size_t SIZE = size_t(1) << 25;  // 32M ints = 128 MB
    std::vector<int> a(SIZE, 1);
    auto stride_pass = [&](size_t stride) {
        long long s = 0;
        for (size_t offset = 0; offset < stride; ++offset)  // every element is touched exactly once
            for (size_t i = offset; i < SIZE; i += stride) s += a[i];
        return s;
    };
    for (size_t stride = 1; stride <= 1024; stride *= 2) {
        sink += stride_pass(stride);  // warm-up (untimed)
        std::vector<double> ns;
        for (int r = 0; r < REPEATS; ++r) {
            auto t0 = Clock::now();
            sink += stride_pass(stride);
            ns.push_back(ms_since(t0) * 1e6 / SIZE);
        }
        Stats st = stats(ns);
        std::printf("stride %5zu ints (%6zu bytes): min %.2f ns/element, median %.2f\n", stride,
                    stride * sizeof(int), st.min, st.median);
    }
    std::printf("(checksum %lld)\n", sink);
}
