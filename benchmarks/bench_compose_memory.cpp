// Memory benchmarks. Two complementary measurements per case:
//   * `bytes_alloc`  - sum of allocations routed through CountingResource
//                      (only library allocations through pmr).
//   * `peak_rss_kb`  - maximum resident set size of the whole process via
//                      getrusage(RUSAGE_SELF).ru_maxrss before/after the run.
//
// CSV export: `--benchmark_format=csv --benchmark_out=memory.csv`.

#include <benchmark/benchmark.h>
#include <sys/resource.h>

#include <algorithm>
#include <memory_resource>
#include <vector>

#include "bench_helpers.hpp"

using pscomp::ModInt998;
using namespace pscomp_bench;

namespace {

std::size_t peak_rss_kb() {
    rusage ru{};
    if (getrusage(RUSAGE_SELF, &ru) != 0) return 0;
#if defined(__APPLE__)
    return static_cast<std::size_t>(ru.ru_maxrss / 1024);  // bytes -> kB
#else
    return static_cast<std::size_t>(ru.ru_maxrss);          // kB on Linux
#endif
}

void register_all() {
    using namespace pscomp::algo;
    auto reg = [](const char* name, auto fn, std::size_t cap) {
        for (auto n : ntt_sizes()) {
            if (n > cap) continue;
            benchmark::RegisterBenchmark(name, [fn](benchmark::State& s) {
                const std::size_t nn = static_cast<std::size_t>(s.range(0));
                auto f = make_random_ntt(nn, false, 5u);
                auto g = make_random_ntt(nn, true,  6u);
                std::pmr::memory_resource* prev = std::pmr::get_default_resource();
                pscomp::CountingResource counter;
                std::pmr::set_default_resource(&counter);
                const std::size_t rss_before = peak_rss_kb();
                for (auto _ : s) {
                    counter.reset();
                    auto out = fn(pscomp::span<const ModInt998>(f),
                                  pscomp::span<const ModInt998>(g), nn);
                    benchmark::DoNotOptimize(out);
                }
                const std::size_t rss_after = peak_rss_kb();
                s.counters["bytes_alloc"]   = static_cast<double>(counter.bytes_allocated());
                s.counters["peak_arena_kB"] = static_cast<double>(counter.high_watermark()) / 1024.0;
                s.counters["peak_rss_kB"]   = static_cast<double>(std::max(rss_before, rss_after));
                std::pmr::set_default_resource(prev);
            })->Arg(static_cast<int>(n))->Unit(benchmark::kMillisecond);
        }
    };

    reg("mem/naive_horner",        [](auto f, auto g, std::size_t n) {
        return compose_naive_horner<ModInt998>(f, g, n); }, 4096);
    reg("mem/brent_kung_basic",    [](auto f, auto g, std::size_t n) {
        return compose_brent_kung_basic<ModInt998>(f, g, n); }, 16384);
    reg("mem/brent_kung_opt",      [](auto f, auto g, std::size_t n) {
        return compose_brent_kung_opt<ModInt998>(f, g, n); }, 65536);
    reg("mem/brent_kung_streaming",[](auto f, auto g, std::size_t n) {
        return compose_brent_kung_streaming<ModInt998>(f, g, n); }, 16384);
    reg("mem/kl_basic",            [](auto f, auto g, std::size_t n) {
        return compose_kl_basic<ModInt998>(f, g, n); }, 4096);
    reg("mem/kl_truncated_mul",    [](auto f, auto g, std::size_t n) {
        return compose_kl_truncated_mul<ModInt998>(f, g, n); }, 16384);
}

}  // namespace

int main(int argc, char** argv) {
    register_all();
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
