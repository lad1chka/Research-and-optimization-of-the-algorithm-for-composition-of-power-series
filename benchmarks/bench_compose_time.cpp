// Time benchmarks per (algorithm, backend, n).

#include <benchmark/benchmark.h>

#include <cstdint>

#include "bench_helpers.hpp"
#include "pscomp/transpose/dual_extraction.hpp"

using pscomp::ModInt998;
using namespace pscomp_bench;

namespace {

void register_all_ntt() {
    using namespace pscomp::algo;
    auto add = [](const char* name, auto fn) {
        for (auto n : ntt_sizes()) {
            benchmark::RegisterBenchmark(name, [fn](benchmark::State& s) {
                const std::size_t nn = static_cast<std::size_t>(s.range(0));
                auto f = make_random_ntt(nn, false, 1u);
                auto g = make_random_ntt(nn, true,  2u);
                for (auto _ : s) {
                    auto out = fn(pscomp::span<const ModInt998>(f),
                                  pscomp::span<const ModInt998>(g), nn);
                    benchmark::DoNotOptimize(out);
                }
                s.SetComplexityN(static_cast<std::int64_t>(nn));
            })->Arg(static_cast<int>(n))->Unit(benchmark::kMillisecond);
        }
    };

    add("ntt/naive_horner", [](auto f, auto g, std::size_t n) {
        return n <= 4096
            ? compose_naive_horner<ModInt998>(f, g, n)
            : std::vector<ModInt998>(n);
    });
    add("ntt/brent_kung_basic", [](auto f, auto g, std::size_t n) {
        return n <= 16384
            ? compose_brent_kung_basic<ModInt998>(f, g, n)
            : std::vector<ModInt998>(n);
    });
    add("ntt/brent_kung_opt", [](auto f, auto g, std::size_t n) {
        return compose_brent_kung_opt<ModInt998>(f, g, n);
    });
    add("ntt/brent_kung_streaming", [](auto f, auto g, std::size_t n) {
        return n <= 16384
            ? compose_brent_kung_streaming<ModInt998>(f, g, n)
            : std::vector<ModInt998>(n);
    });
    add("ntt/brent_kung_tuned_m", [](auto f, auto g, std::size_t n) {
        return compose_brent_kung_tuned_m<ModInt998>(f, g, n);
    });
    add("ntt/kl_basic", [](auto f, auto g, std::size_t n) {
        return n <= 4096
            ? compose_kl_basic<ModInt998>(f, g, n)
            : std::vector<ModInt998>(n);
    });
    add("ntt/kl_truncated_mul", [](auto f, auto g, std::size_t n) {
        return n <= 16384
            ? compose_kl_truncated_mul<ModInt998>(f, g, n)
            : std::vector<ModInt998>(n);
    });
    add("ntt/kl_threshold", [](auto f, auto g, std::size_t n) {
        return n <= 16384
            ? compose_kl_recursion_threshold<ModInt998>(f, g, n, 64)
            : std::vector<ModInt998>(n);
    });
    add("ntt/kl_tellegen", [](auto f, auto g, std::size_t n) {
        return n <= 4096
            ? compose_kl_tellegen<ModInt998>(f, g, n)
            : std::vector<ModInt998>(n);
    });

    using pscomp::transpose::extract_dual_basic;
    using pscomp::transpose::extract_dual_inplace;
    using pscomp::transpose::extract_dual_truncated_mul;
    using pscomp::transpose::extract_dual_threshold;

    add("ntt/kl_dual_basic", [](auto f, auto g, std::size_t n) {
        return n <= 16384
            ? extract_dual_basic<ModInt998>(f, g, n, n)
            : std::vector<ModInt998>(n);
    });
    add("ntt/kl_dual_inplace", [](auto f, auto g, std::size_t n) {
        return n <= 16384
            ? extract_dual_inplace<ModInt998>(f, g, n, n)
            : std::vector<ModInt998>(n);
    });
    add("ntt/kl_dual_truncated_mul", [](auto f, auto g, std::size_t n) {
        return n <= 65536
            ? extract_dual_truncated_mul<ModInt998>(f, g, n, n)
            : std::vector<ModInt998>(n);
    });
    add("ntt/kl_dual_threshold", [](auto f, auto g, std::size_t n) {
        return n <= 65536
            ? extract_dual_threshold<ModInt998>(f, g, n, n, 64)
            : std::vector<ModInt998>(n);
    });
}

void register_all_fft() {
    using namespace pscomp::algo;
    auto add = [](const char* name, auto fn) {
        for (auto n : fft_sizes()) {
            benchmark::RegisterBenchmark(name, [fn](benchmark::State& s) {
                const std::size_t nn = static_cast<std::size_t>(s.range(0));
                auto f = make_random_real<double>(nn, false, 3u);
                auto g = make_random_real<double>(nn, true,  4u);
                for (auto _ : s) {
                    auto out = fn(pscomp::span<const double>(f),
                                  pscomp::span<const double>(g), nn);
                    benchmark::DoNotOptimize(out);
                }
                s.SetComplexityN(static_cast<std::int64_t>(nn));
            })->Arg(static_cast<int>(n))->Unit(benchmark::kMillisecond);
        }
    };

    add("fft/naive_horner", [](auto f, auto g, std::size_t n) {
        return n <= 4096
            ? compose_naive_horner<double>(f, g, n)
            : std::vector<double>(n);
    });
    add("fft/brent_kung_opt", [](auto f, auto g, std::size_t n) {
        return compose_brent_kung_opt<double>(f, g, n);
    });
    add("fft/kl_truncated_mul", [](auto f, auto g, std::size_t n) {
        return n <= 16384
            ? compose_kl_truncated_mul<double>(f, g, n)
            : std::vector<double>(n);
    });
    add("fft/kl_tellegen", [](auto f, auto g, std::size_t n) {
        return n <= 4096
            ? compose_kl_tellegen<double>(f, g, n)
            : std::vector<double>(n);
    });
}

}  // namespace

int main(int argc, char** argv) {
    register_all_ntt();
    register_all_fft();
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
