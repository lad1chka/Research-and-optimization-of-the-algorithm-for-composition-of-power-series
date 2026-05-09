// Middle product: out[k] = sum_{j=0}^{m-1} b[j] * x[k+j], the transpose of
// a -> a*b with b fixed.

#ifndef PSCOMP_TRANSPOSE_MIDDLE_PRODUCT_HPP
#define PSCOMP_TRANSPOSE_MIDDLE_PRODUCT_HPP

#include <algorithm>
#include <cstddef>
#include <type_traits>
#include <vector>

#include "pscomp/field/coef_traits.hpp"
#include "pscomp/field/mod_int.hpp"
#include "pscomp/poly/poly_mul.hpp"
#include "pscomp/span.hpp"
#include "pscomp/transform/fft.hpp"
#include "pscomp/transform/ntt.hpp"
#include "pscomp/transform/ntt_basic.hpp"

namespace pscomp::transpose {

template <class Coef>
std::vector<Coef> mp_schoolbook(span<const Coef> b,
                                span<const Coef> x,
                                std::size_t n_out) {
    std::vector<Coef> out(n_out, coef_zero<Coef>());
    if (b.empty() || n_out == 0) return out;
    const std::size_t m = b.size();
    for (std::size_t k = 0; k < n_out; ++k) {
        Coef acc = coef_zero<Coef>();
        const std::size_t lim = std::min(m, x.size() > k ? x.size() - k : 0);
        for (std::size_t j = 0; j < lim; ++j) {
            acc = acc + b[j] * x[k + j];
        }
        out[k] = acc;
    }
    return out;
}

namespace detail {

// rev(b) * x, then take coefficients [m-1 .. m+n_out-2].
template <class Coef, class Mul>
std::vector<Coef> mp_via_convolution(span<const Coef> b,
                                     span<const Coef> x,
                                     std::size_t n_out,
                                     Mul&& mul) {
    std::vector<Coef> out(n_out, coef_zero<Coef>());
    if (b.empty() || x.empty() || n_out == 0) return out;
    const std::size_t m = b.size();

    std::vector<Coef> rb(m);
    for (std::size_t i = 0; i < m; ++i) rb[i] = b[m - 1 - i];

    auto full = mul(span<const Coef>(rb.data(), rb.size()), x);
    const std::size_t base = m - 1;
    for (std::size_t t = 0; t < n_out && base + t < full.size(); ++t) {
        out[t] = full[base + t];
    }
    return out;
}

}  // namespace detail

template <class Real,
          class = std::enable_if_t<is_real_v<Real>>>
std::vector<Real> mp_fft(span<const Real> b,
                         span<const Real> x,
                         std::size_t n_out) {
    return detail::mp_via_convolution<Real>(b, x, n_out,
        [](span<const Real> u, span<const Real> v) {
            return fft::multiply<Real>(u, v);
        });
}

inline std::vector<ModInt998> mp_ntt(span<const ModInt998> b,
                                     span<const ModInt998> x,
                                     std::size_t n_out) {
    return detail::mp_via_convolution<ModInt998>(b, x, n_out,
        [](span<const ModInt998> u, span<const ModInt998> v) {
            return ntt::multiply(u, v);
        });
}

inline std::vector<ModInt998Plain> mp_ntt_basic(span<const ModInt998Plain> b,
                                                span<const ModInt998Plain> x,
                                                std::size_t n_out) {
    return detail::mp_via_convolution<ModInt998Plain>(b, x, n_out,
        [](span<const ModInt998Plain> u, span<const ModInt998Plain> v) {
            return ntt_basic::multiply(u, v);
        });
}

template <class Coef>
std::vector<Coef> middle_product(span<const Coef> b,
                                 span<const Coef> x,
                                 std::size_t n_out) {
    if constexpr (std::is_same_v<Coef, ModInt998>) {
        return mp_ntt(b, x, n_out);
    } else if constexpr (std::is_same_v<Coef, ModInt998Plain>) {
        return mp_ntt_basic(b, x, n_out);
    } else if constexpr (is_real_v<Coef>) {
        return mp_fft<Coef>(b, x, n_out);
    } else {
        return mp_schoolbook<Coef>(b, x, n_out);
    }
}

}  // namespace pscomp::transpose

#endif
