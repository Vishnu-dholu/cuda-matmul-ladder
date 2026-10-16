// cache_sim.cpp
// Build: g++ -O2 -std=c++17 cache_sim.cpp -o cache_sim
// Run:   ./cache_sim 32768 64 8 row.trace     (cache bytes, line bytes, ways, trace file)
// Trace: one hexadecimal byte address per line.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

struct Line {
    bool valid = false;
    uint64_t tag = 0;
    uint64_t last_used = 0;  // time stamp used for LRU
};

static unsigned ilog2(uint64_t x) {
    unsigned r = 0;
    while (x > 1) {
        x >>= 1;
        ++r;
    }
    return r;
}
static bool is_pow2(uint64_t x) { return x && !(x & (x - 1)); }

class Cache {
   public:
    Cache(uint64_t cache_bytes, uint64_t line_bytes, uint64_t ways) : ways_(ways) {
        num_sets_ = cache_bytes / (line_bytes * ways);
        if (!is_pow2(line_bytes) || !is_pow2(num_sets_)) {
            std::fprintf(stderr, "line size and number of sets must be powers of two\n");
            std::exit(1);
        }
        offset_bits_ = ilog2(line_bytes);
        index_bits_ = ilog2(num_sets_);
        sets_.assign(num_sets_, std::vector<Line>(ways_));
    }

    bool access(uint64_t addr) {  // returns true on a hit
        ++clock_;
        uint64_t index = (addr >> offset_bits_) & (num_sets_ - 1);
        uint64_t tag = addr >> (offset_bits_ + index_bits_);
        auto& set = sets_[index];

        for (auto& l : set)  // 1) look for a hit
            if (l.valid && l.tag == tag) {
                l.last_used = clock_;
                ++hits_;
                return true;
            }

        ++misses_;  // 2) miss: choose a victim
        Line* victim = &set[0];
        for (auto& l : set) {
            if (!l.valid) {
                victim = &l;
                break;
            }                                                  // free way first
            if (l.last_used < victim->last_used) victim = &l;  // else least recently used
        }
        victim->valid = true;
        victim->tag = tag;
        victim->last_used = clock_;
        return false;
    }

    void report() const {
        uint64_t total = hits_ + misses_;
        std::printf("sets=%llu ways=%llu | accesses=%llu hits=%llu misses=%llu | hit rate=%.2f%%\n",
                    (unsigned long long)num_sets_, (unsigned long long)ways_,
                    (unsigned long long)total, (unsigned long long)hits_,
                    (unsigned long long)misses_, total ? 100.0 * hits_ / total : 0.0);
    }

   private:
    uint64_t num_sets_ = 0, ways_, clock_ = 0, hits_ = 0, misses_ = 0;
    unsigned offset_bits_ = 0, index_bits_ = 0;
    std::vector<std::vector<Line>> sets_;
};

int main(int argc, char** argv) {
    if (argc != 5) {
        std::fprintf(stderr, "usage: %s cache_bytes line_bytes ways trace_file\n", argv[0]);
        return 1;
    }
    Cache cache(std::stoull(argv[1]), std::stoull(argv[2]), std::stoull(argv[3]));
    std::ifstream in(argv[4]);
    std::string s;
    while (in >> s) cache.access(std::stoull(s, nullptr, 16));
    cache.report();
}
