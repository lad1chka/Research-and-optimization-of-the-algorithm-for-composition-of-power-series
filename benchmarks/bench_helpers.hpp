// Shared inputs and size lists for the benchmark binaries.

#ifndef PSCOMP_BENCH_HELPERS_HPP
#define PSCOMP_BENCH_HELPERS_HPP

#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "pscomp/pscomp.hpp"

namespace pscomp_bench {

constexpr std::uint32_t kSeed = 42;

inline std::vector<pscomp::ModInt998> make_random_ntt(std::size_t n,
                                                       bool g0_zero = false,
                                                       std::uint32_t salt = 0) {
    std::mt19937 rng(kSeed ^ static_cast<std::uint32_t>(n) ^ salt);
    std::uniform_int_distribution<std::uint32_t> dist(0u, pscomp::kPrime998 - 1u);
    std::vector<pscomp::ModInt998> v(n);
    for (std::size_t i = 0; i < n; ++i) v[i] = pscomp::ModInt998{dist(rng)};
    if (g0_zero && !v.empty()) v[0] = pscomp::ModInt998{};
    return v;
}

template <class Real>
std::vector<Real> make_random_real(std::size_t n,
                                   bool g0_zero = false,
                                   std::uint32_t salt = 0) {
    std::mt19937 rng(kSeed ^ static_cast<std::uint32_t>(n) ^ salt);
    std::uniform_real_distribution<Real> dist(static_cast<Real>(-0.5),
                                              static_cast<Real>(0.5));
    std::vector<Real> v(n);
    for (std::size_t i = 0; i < n; ++i) v[i] = dist(rng);
    if (g0_zero && !v.empty()) v[0] = Real(0);
    return v;
}

inline const std::vector<std::size_t>& ntt_sizes() {
    static const std::vector<std::size_t> v = {256, 1024, 4096, 16384, 65536};
    return v;
}

inline const std::vector<std::size_t>& fft_sizes() {
    static const std::vector<std::size_t> v = {256, 1024, 4096, 16384, 65536};
    return v;
}

}  // namespace pscomp_bench

#endif
