// Truncated-power-series helpers (resize, truncate, reverse, pow2 round-up).

#ifndef PSCOMP_POLY_POLY_UTILS_HPP
#define PSCOMP_POLY_POLY_UTILS_HPP

#include <algorithm>
#include <cstddef>
#include <vector>

#include "pscomp/field/coef_traits.hpp"
#include "pscomp/span.hpp"

namespace pscomp::poly {

template <class Coef>
std::vector<Coef> resized(span<const Coef> a, std::size_t n) {
    std::vector<Coef> out(n, coef_zero<Coef>());
    const std::size_t k = std::min(n, a.size());
    for (std::size_t i = 0; i < k; ++i) out[i] = a[i];
    return out;
}

template <class Coef>
void truncate_inplace(std::vector<Coef>& a, std::size_t n) {
    a.resize(n, coef_zero<Coef>());
}

template <class Coef>
std::vector<Coef> reversed(span<const Coef> a) {
    std::vector<Coef> out(a.begin(), a.end());
    std::reverse(out.begin(), out.end());
    return out;
}

inline std::size_t round_up_pow2(std::size_t n) noexcept {
    if (n <= 1) return 1;
    std::size_t r = 1;
    while (r < n) r <<= 1;
    return r;
}

}  // namespace pscomp::poly

#endif
