// Iterative NTT over 998244353 (Montgomery arithmetic via ModInt998).

#ifndef PSCOMP_TRANSFORM_NTT_HPP
#define PSCOMP_TRANSFORM_NTT_HPP

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include "pscomp/field/mod_int.hpp"
#include "pscomp/span.hpp"

namespace pscomp::ntt {

namespace detail {

inline std::uint32_t log2_floor(std::uint32_t n) noexcept {
#if defined(__GNUC__) || defined(__clang__)
    return 31u - static_cast<std::uint32_t>(__builtin_clz(n));
#else
    std::uint32_t r = 0;
    while ((1u << (r + 1)) <= n) ++r;
    return r;
#endif
}

inline const std::vector<std::uint32_t>& bitrev_cache(std::uint32_t n) {
    thread_local std::unordered_map<std::uint32_t, std::vector<std::uint32_t>> c;
    auto& v = c[n];
    if (v.size() == n) return v;
    v.assign(n, 0);
    const std::uint32_t logn = log2_floor(n);
    for (std::uint32_t i = 0; i < n; ++i) {
        v[i] = (v[i >> 1] >> 1) | ((i & 1u) << (logn - 1));
    }
    return v;
}

inline const std::vector<ModInt998>& twiddles(std::uint32_t n, bool inverse) {
    thread_local std::unordered_map<std::uint32_t, std::vector<ModInt998>> fwd, inv;
    auto& bucket = inverse ? inv : fwd;
    auto& v = bucket[n];
    if (v.size() == n) return v;
    v.assign(n, ModInt998{1u});
    ModInt998 g = ModInt998{kPrime998Root}.pow((kPrime998 - 1u) / n);
    if (inverse) g = g.inv();
    ModInt998 cur{1u};
    for (std::uint32_t i = 0; i < n; ++i) {
        v[i] = cur;
        cur *= g;
    }
    return v;
}

}  // namespace detail

inline void transform(span<ModInt998> a, bool inverse) {
    const std::size_t n = a.size();
    if (n <= 1) return;
    assert((n & (n - 1)) == 0 && "ntt size must be a power of two");

    const auto& rev = detail::bitrev_cache(static_cast<std::uint32_t>(n));
    for (std::size_t i = 0; i < n; ++i) {
        if (i < rev[i]) std::swap(a[i], a[rev[i]]);
    }

    const auto& tw = detail::twiddles(static_cast<std::uint32_t>(n), inverse);
    for (std::size_t len = 2; len <= n; len <<= 1) {
        const std::size_t half = len >> 1;
        const std::size_t step = n / len;
        for (std::size_t base = 0; base < n; base += len) {
            for (std::size_t j = 0; j < half; ++j) {
                const ModInt998 u = a[base + j];
                const ModInt998 v = a[base + j + half] * tw[j * step];
                a[base + j]        = u + v;
                a[base + j + half] = u - v;
            }
        }
    }

    if (inverse) {
        const ModInt998 inv_n = ModInt998{static_cast<std::uint64_t>(n)}.inv();
        for (auto& x : a) x *= inv_n;
    }
}

inline std::vector<ModInt998> multiply(span<const ModInt998> a,
                                       span<const ModInt998> b) {
    if (a.empty() || b.empty()) return {};
    const std::size_t result_size = a.size() + b.size() - 1;
    std::size_t n = 1;
    while (n < result_size) n <<= 1;

    std::vector<ModInt998> ca(n), cb(n);
    for (std::size_t i = 0; i < a.size(); ++i) ca[i] = a[i];
    for (std::size_t i = 0; i < b.size(); ++i) cb[i] = b[i];
    transform(span<ModInt998>(ca.data(), n), false);
    transform(span<ModInt998>(cb.data(), n), false);
    for (std::size_t i = 0; i < n; ++i) ca[i] *= cb[i];
    transform(span<ModInt998>(ca.data(), n), true);
    ca.resize(result_size);
    return ca;
}

}  // namespace pscomp::ntt

#endif
