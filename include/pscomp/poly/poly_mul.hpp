// Polynomial multiplication backends and a coefficient-type dispatcher.

#ifndef PSCOMP_POLY_POLY_MUL_HPP
#define PSCOMP_POLY_POLY_MUL_HPP

#include <complex>
#include <cstddef>
#include <vector>

#include "pscomp/field/coef_traits.hpp"
#include "pscomp/field/mod_int.hpp"
#include "pscomp/poly/poly_utils.hpp"
#include "pscomp/span.hpp"
#include "pscomp/transform/fft.hpp"
#include "pscomp/transform/fft_basic.hpp"
#include "pscomp/transform/ntt.hpp"
#include "pscomp/transform/ntt_basic.hpp"

namespace pscomp::poly {

template <class Coef>
std::vector<Coef> mul_schoolbook(span<const Coef> a, span<const Coef> b) {
    if (a.empty() || b.empty()) return {};
    std::vector<Coef> out(a.size() + b.size() - 1, coef_zero<Coef>());
    for (std::size_t i = 0; i < a.size(); ++i) {
        const Coef ai = a[i];
        for (std::size_t j = 0; j < b.size(); ++j) {
            out[i + j] = out[i + j] + ai * b[j];
        }
    }
    return out;
}

template <class Real,
          class = std::enable_if_t<is_real_v<Real>>>
std::vector<Real> mul_fft(span<const Real> a, span<const Real> b) {
    return fft::multiply<Real>(a, b);
}

template <class Real,
          class = std::enable_if_t<is_real_v<Real>>>
std::vector<Real> mul_fft_basic(span<const Real> a, span<const Real> b) {
    return fft_basic::multiply<Real>(a, b);
}

inline std::vector<ModInt998> mul_ntt(span<const ModInt998> a,
                                      span<const ModInt998> b) {
    return ntt::multiply(a, b);
}

inline std::vector<ModInt998Plain> mul_ntt_basic(span<const ModInt998Plain> a,
                                                 span<const ModInt998Plain> b) {
    return ntt_basic::multiply(a, b);
}

template <class Coef>
std::vector<Coef> mul(span<const Coef> a, span<const Coef> b) {
    if constexpr (std::is_same_v<Coef, ModInt998>) {
        return mul_ntt(a, b);
    } else if constexpr (std::is_same_v<Coef, ModInt998Plain>) {
        return mul_ntt_basic(a, b);
    } else if constexpr (is_real_v<Coef>) {
        return mul_fft<Coef>(a, b);
    } else {
        return mul_schoolbook<Coef>(a, b);
    }
}

template <class Coef>
std::vector<Coef> mul_truncated(span<const Coef> a, span<const Coef> b,
                                std::size_t n) {
    auto full = mul<Coef>(a, b);
    full.resize(n, coef_zero<Coef>());
    return full;
}

}  // namespace pscomp::poly

#endif
