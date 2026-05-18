// Memory benchmarks: peak heap usage via global operator new/delete override.

#include <benchmark/benchmark.h>

#include <vector>

#include "bench_helpers.hpp"
#include "pscomp/workspace/heap_tracker.hpp"
#include "pscomp/transpose/dual_extraction.hpp"

using pscomp::ModInt998;
using pscomp::HeapTracker;
using namespace pscomp_bench;

namespace {

void register_all() {
    using namespace pscomp::algo;
    auto reg = [](const char* name, auto fn, std::size_t cap) {
        for (auto n : ntt_sizes()) {
            if (n > cap) continue;
            benchmark::RegisterBenchmark(name, [fn](benchmark::State& s) {
                const std::size_t nn = static_cast<std::size_t>(s.range(0));
                auto f = make_random_ntt(nn, false, 5u);
                auto g = make_random_ntt(nn, true,  6u);
                for (auto _ : s) {
                    HeapTracker::active = true;
                    HeapTracker::reset();
                    auto out = fn(pscomp::span<const ModInt998>(f),
                                  pscomp::span<const ModInt998>(g), nn);
                    benchmark::DoNotOptimize(out);
                    HeapTracker::active = false;
                }
                s.counters["peak_heap_kB"] =
                    static_cast<double>(HeapTracker::peak) / 1024.0;
            })->Arg(static_cast<int>(n))->Unit(benchmark::kMillisecond);
        }
    };

    reg("mem/naive_horner",        [](auto f, auto g, std::size_t n) {
        return compose_naive_horner<ModInt998>(f, g, n); }, 4096);
    reg("mem/brent_kung_basic",    [](auto f, auto g, std::size_t n) {
        return compose_brent_kung_basic<ModInt998>(f, g, n); }, 65536);
    reg("mem/brent_kung_opt",      [](auto f, auto g, std::size_t n) {
        return compose_brent_kung_opt<ModInt998>(f, g, n); }, 65536);
    reg("mem/brent_kung_streaming",[](auto f, auto g, std::size_t n) {
        return compose_brent_kung_streaming<ModInt998>(f, g, n); }, 65536);
    reg("mem/brent_kung_tuned_m",  [](auto f, auto g, std::size_t n) {
        return compose_brent_kung_tuned_m<ModInt998>(f, g, n); }, 65536);
    reg("mem/kl_basic",            [](auto f, auto g, std::size_t n) {
        return compose_kl_basic<ModInt998>(f, g, n); }, 4096);
    reg("mem/kl_truncated_mul",    [](auto f, auto g, std::size_t n) {
        return compose_kl_truncated_mul<ModInt998>(f, g, n); }, 65536);
    reg("mem/kl_threshold",        [](auto f, auto g, std::size_t n) {
        return compose_kl_recursion_threshold<ModInt998>(f, g, n, 64); }, 65536);
    reg("mem/kl_tellegen",         [](auto f, auto g, std::size_t n) {
        return compose_kl_tellegen<ModInt998>(f, g, n); }, 4096);

    using pscomp::transpose::extract_dual_basic;
    using pscomp::transpose::extract_dual_inplace;
    using pscomp::transpose::extract_dual_truncated_mul;
    using pscomp::transpose::extract_dual_threshold;
    using pscomp::transpose::extract_dual_combined;

    reg("mem/kl_dual_basic",         [](auto f, auto g, std::size_t n) {
        return extract_dual_basic<ModInt998>(f, g, n, n); }, 65536);
    reg("mem/kl_dual_inplace",       [](auto f, auto g, std::size_t n) {
        return extract_dual_inplace<ModInt998>(f, g, n, n); }, 65536);
    reg("mem/kl_dual_truncated_mul", [](auto f, auto g, std::size_t n) {
        return extract_dual_truncated_mul<ModInt998>(f, g, n, n); }, 65536);
    reg("mem/kl_dual_threshold",     [](auto f, auto g, std::size_t n) {
        return extract_dual_threshold<ModInt998>(f, g, n, n, 64); }, 65536);
    reg("mem/kl_dual_combined",      [](auto f, auto g, std::size_t n) {
        return extract_dual_combined<ModInt998>(f, g, n, n, 64); }, 65536);
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
