// Iterative Cooley-Tukey FFT with cached twiddles and bit-reverse tables.

#ifndef PSCOMP_TRANSFORM_FFT_HPP
#define PSCOMP_TRANSFORM_FFT_HPP

#include <cassert>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include "pscomp/span.hpp"

namespace pscomp::fft {

namespace detail {

template <class Real>
constexpr Real kPi = static_cast<Real>(3.141592653589793238462643383279502884L);

inline std::uint32_t log2_floor(std::uint32_t n) noexcept {
#if defined(__GNUC__) || defined(__clang__)
    return 31u - static_cast<std::uint32_t>(__builtin_clz(n));
#else
    std::uint32_t r = 0;
    while ((1u << (r + 1)) <= n) ++r;
    return r;
#endif
}

template <class Real>
struct TwiddleCache {
    std::unordered_map<std::uint32_t, std::vector<std::complex<Real>>> fwd;
    std::unordered_map<std::uint32_t, std::vector<std::complex<Real>>> inv;
};

template <class Real>
TwiddleCache<Real>& twiddle_cache() {
    thread_local TwiddleCache<Real> c;
    return c;
}

inline std::vector<std::uint32_t>& bitrev_cache(std::uint32_t n) {
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

template <class Real>
const std::vector<std::complex<Real>>& twiddles(std::uint32_t n, bool inverse) {
    auto& cache = twiddle_cache<Real>();
    auto& bucket = inverse ? cache.inv : cache.fwd;
    auto& v = bucket[n];
    if (v.size() == n) return v;
    v.assign(n, std::complex<Real>(1, 0));
    const Real ang = (inverse ? -2 : 2) * detail::kPi<Real> / static_cast<Real>(n);
    for (std::uint32_t i = 0; i < n; ++i) {
        v[i] = std::complex<Real>(std::cos(ang * i), std::sin(ang * i));
    }
    return v;
}

}  // namespace detail

template <class Real>
void transform(span<std::complex<Real>> a, bool inverse) {
    const std::size_t n = a.size();
    if (n <= 1) return;
    assert((n & (n - 1)) == 0 && "fft size must be a power of two");

    const auto& rev = detail::bitrev_cache(static_cast<std::uint32_t>(n));
    for (std::size_t i = 0; i < n; ++i) {
        if (i < rev[i]) std::swap(a[i], a[rev[i]]);
    }

    const auto& tw = detail::twiddles<Real>(static_cast<std::uint32_t>(n), inverse);
    for (std::size_t len = 2; len <= n; len <<= 1) {
        const std::size_t half = len >> 1;
        const std::size_t step = n / len;
        for (std::size_t base = 0; base < n; base += len) {
            for (std::size_t j = 0; j < half; ++j) {
                const std::complex<Real> u = a[base + j];
                const std::complex<Real> v = a[base + j + half] * tw[j * step];
                a[base + j]        = u + v;
                a[base + j + half] = u - v;
            }
        }
    }

    if (inverse) {
        const Real inv_n = Real(1) / static_cast<Real>(n);
        for (auto& z : a) z *= inv_n;
    }
}

template <class Real>
std::vector<Real> multiply(span<const Real> a, span<const Real> b) {
    if (a.empty() || b.empty()) return {};
    const std::size_t result_size = a.size() + b.size() - 1;
    std::size_t n = 1;
    while (n < result_size) n <<= 1;

    std::vector<std::complex<Real>> ca(n), cb(n);
    for (std::size_t i = 0; i < a.size(); ++i) ca[i] = std::complex<Real>(a[i], 0);
    for (std::size_t i = 0; i < b.size(); ++i) cb[i] = std::complex<Real>(b[i], 0);
    transform<Real>(ca, false);
    transform<Real>(cb, false);
    for (std::size_t i = 0; i < n; ++i) ca[i] *= cb[i];
    transform<Real>(ca, true);

    std::vector<Real> out(result_size);
    for (std::size_t i = 0; i < result_size; ++i) out[i] = ca[i].real();
    return out;
}

// Two real inputs packed into the real/imag parts of one complex transform.
template <class Real>
void packed_real_forward(span<const Real> a, span<const Real> b,
                         std::vector<std::complex<Real>>& spec_a,
                         std::vector<std::complex<Real>>& spec_b) {
    const std::size_t n = std::max(a.size(), b.size());
    std::size_t m = 1;
    while (m < n) m <<= 1;
    std::vector<std::complex<Real>> z(m, std::complex<Real>(0, 0));
    for (std::size_t i = 0; i < a.size(); ++i) z[i].real(a[i]);
    for (std::size_t i = 0; i < b.size(); ++i) z[i].imag(b[i]);
    transform<Real>(z, false);

    spec_a.assign(m, std::complex<Real>(0, 0));
    spec_b.assign(m, std::complex<Real>(0, 0));
    for (std::size_t i = 0; i < m; ++i) {
        const std::size_t j = (m - i) & (m - 1);
        const std::complex<Real> p = z[i];
        const std::complex<Real> q = std::conj(z[j]);
        spec_a[i] = (p + q) * Real(0.5);
        spec_b[i] = std::complex<Real>(0, -Real(0.5)) * (p - q);
    }
}

}  // namespace pscomp::fft

#endif
