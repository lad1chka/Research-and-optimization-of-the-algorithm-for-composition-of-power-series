// Reference recursive NTT over the prime 998244353. Operates on ModInt998Plain
// (straight `% P` arithmetic) so that benchmarks can isolate three combined
// effects against the optimized variant: recursive vs iterative form,
// per-call twiddle generation vs cache, and `% P` vs Montgomery multiplication.

#ifndef PSCOMP_TRANSFORM_NTT_BASIC_HPP
#define PSCOMP_TRANSFORM_NTT_BASIC_HPP

#include <cassert>
#include <cstddef>
#include <vector>

#include "pscomp/field/mod_int.hpp"
#include "pscomp/span.hpp"

namespace pscomp::ntt_basic {

inline void transform(span<ModInt998Plain> a, bool inverse) {
    const std::size_t n = a.size();
    if (n <= 1) return;
    assert((n & (n - 1)) == 0 && "ntt size must be a power of two");

    const std::size_t half = n / 2;
    std::vector<ModInt998Plain> evens(half), odds(half);
    for (std::size_t i = 0; i < half; ++i) {
        evens[i] = a[2 * i];
        odds[i]  = a[2 * i + 1];
    }
    transform(span<ModInt998Plain>(evens.data(), half), inverse);
    transform(span<ModInt998Plain>(odds.data(),  half), inverse);

    ModInt998Plain g = ModInt998Plain{kPrime998Root}.pow((kPrime998 - 1) / n);
    if (inverse) g = g.inv();
    ModInt998Plain w{1u};
    for (std::size_t i = 0; i < half; ++i) {
        const ModInt998Plain t = w * odds[i];
        a[i]        = evens[i] + t;
        a[i + half] = evens[i] - t;
        w *= g;
    }
}

inline std::vector<ModInt998Plain> multiply(span<const ModInt998Plain> a,
                                            span<const ModInt998Plain> b) {
    if (a.empty() || b.empty()) return {};
    const std::size_t result_size = a.size() + b.size() - 1;
    std::size_t n = 1;
    while (n < result_size) n <<= 1;
    std::vector<ModInt998Plain> ca(n), cb(n);
    for (std::size_t i = 0; i < a.size(); ++i) ca[i] = a[i];
    for (std::size_t i = 0; i < b.size(); ++i) cb[i] = b[i];
    transform(span<ModInt998Plain>(ca.data(), n), false);
    transform(span<ModInt998Plain>(cb.data(), n), false);
    for (std::size_t i = 0; i < n; ++i) ca[i] = ca[i] * cb[i];
    transform(span<ModInt998Plain>(ca.data(), n), true);
    const ModInt998Plain inv_n = ModInt998Plain{static_cast<std::uint64_t>(n)}.inv();
    for (auto& x : ca) x = x * inv_n;
    ca.resize(result_size);
    return ca;
}

}  // namespace pscomp::ntt_basic

#endif  // PSCOMP_TRANSFORM_NTT_BASIC_HPP
