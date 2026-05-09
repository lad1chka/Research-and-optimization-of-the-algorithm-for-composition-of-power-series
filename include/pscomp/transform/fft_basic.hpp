// Recursive Cooley-Tukey FFT (reference implementation).

#ifndef PSCOMP_TRANSFORM_FFT_BASIC_HPP
#define PSCOMP_TRANSFORM_FFT_BASIC_HPP

#include <cassert>
#include <cmath>
#include <complex>
#include <cstddef>
#include <vector>

#include "pscomp/span.hpp"

namespace pscomp::fft_basic {

namespace detail {
template <class Real>
constexpr Real kPi = static_cast<Real>(3.141592653589793238462643383279502884L);
}

template <class Real>
void transform(span<std::complex<Real>> a, bool inverse) {
    const std::size_t n = a.size();
    if (n <= 1) return;
    assert((n & (n - 1)) == 0 && "fft size must be a power of two");

    const std::size_t half = n / 2;
    std::vector<std::complex<Real>> evens(half), odds(half);
    for (std::size_t i = 0; i < half; ++i) {
        evens[i] = a[2 * i];
        odds[i]  = a[2 * i + 1];
    }
    transform<Real>(span<std::complex<Real>>(evens.data(), half), inverse);
    transform<Real>(span<std::complex<Real>>(odds.data(),  half), inverse);

    const Real ang = (inverse ? -2 : 2) * detail::kPi<Real> / static_cast<Real>(n);
    std::complex<Real> w(1), wn(std::cos(ang), std::sin(ang));
    for (std::size_t i = 0; i < half; ++i) {
        const std::complex<Real> t = w * odds[i];
        a[i]        = evens[i] + t;
        a[i + half] = evens[i] - t;
        w *= wn;
    }
}

template <class Real>
std::vector<std::complex<Real>> forward(span<const std::complex<Real>> in) {
    std::vector<std::complex<Real>> out(in.begin(), in.end());
    transform<Real>(span<std::complex<Real>>(out.data(), out.size()), false);
    return out;
}

template <class Real>
std::vector<std::complex<Real>> inverse(span<const std::complex<Real>> in) {
    std::vector<std::complex<Real>> out(in.begin(), in.end());
    transform<Real>(span<std::complex<Real>>(out.data(), out.size()), true);
    const Real inv_n = Real(1) / static_cast<Real>(out.size());
    for (auto& z : out) z *= inv_n;
    return out;
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
    transform<Real>(span<std::complex<Real>>(ca.data(), n), false);
    transform<Real>(span<std::complex<Real>>(cb.data(), n), false);
    for (std::size_t i = 0; i < n; ++i) ca[i] *= cb[i];
    transform<Real>(span<std::complex<Real>>(ca.data(), n), true);

    const Real inv_n = Real(1) / static_cast<Real>(n);
    std::vector<Real> out(result_size);
    for (std::size_t i = 0; i < result_size; ++i) out[i] = ca[i].real() * inv_n;
    return out;
}

}  // namespace pscomp::fft_basic

#endif
