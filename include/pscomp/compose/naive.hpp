// Naive composition baselines.
//
//   compose_naive_def      - O(n^3) literal sum_j a_j g(x)^j evaluation.
//   compose_naive_horner   - O(n^2) Horner scheme with mod x^n truncation.
//   compose_naive_horner_inplace - same complexity but writes into a caller-
//                                  provided workspace and avoids per-call
//                                  std::vector allocations.

#ifndef PSCOMP_COMPOSE_NAIVE_HPP
#define PSCOMP_COMPOSE_NAIVE_HPP

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "pscomp/field/coef_traits.hpp"
#include "pscomp/poly/poly_mul.hpp"
#include "pscomp/poly/poly_utils.hpp"
#include "pscomp/span.hpp"

namespace pscomp::algo {

namespace detail {
template <class Coef>
void require_g0_zero(span<const Coef> g) {
    if (!g.empty() && g[0] != coef_zero<Coef>()) {
#ifdef NDEBUG
        throw std::invalid_argument("pscomp::compose: g(0) must be 0");
#else
        assert(false && "pscomp::compose: g(0) must be 0");
#endif
    }
}
}  // namespace detail

template <class Coef>
std::vector<Coef> compose_naive_def(span<const Coef> f,
                                    span<const Coef> g,
                                    std::size_t n) {
    detail::require_g0_zero(g);
    std::vector<Coef> out(n, coef_zero<Coef>());
    if (n == 0 || f.empty()) return out;

    // power = g^0 = 1.
    std::vector<Coef> power(n, coef_zero<Coef>());
    power[0] = coef_one<Coef>();

    const std::size_t fn = std::min<std::size_t>(f.size(), n);
    std::vector<Coef> g_trunc = poly::resized<Coef>(g, n);

    for (std::size_t j = 0; j < fn; ++j) {
        const Coef aj = f[j];
        if (aj != coef_zero<Coef>()) {
            for (std::size_t i = 0; i < n; ++i) out[i] = out[i] + aj * power[i];
        }
        if (j + 1 < fn) {
            // power := power * g_trunc, truncated to n.
            auto next = poly::mul_truncated<Coef>(
                span<const Coef>(power.data(), power.size()),
                span<const Coef>(g_trunc.data(), g_trunc.size()), n);
            power.swap(next);
        }
    }
    return out;
}

template <class Coef>
std::vector<Coef> compose_naive_horner(span<const Coef> f,
                                       span<const Coef> g,
                                       std::size_t n) {
    detail::require_g0_zero(g);
    if (n == 0) return {};

    std::vector<Coef> out(n, coef_zero<Coef>());
    if (f.empty()) return out;

    std::vector<Coef> g_trunc = poly::resized<Coef>(g, n);
    out[0] = f[f.size() - 1];

    for (std::size_t k = f.size(); k-- > 0;) {
        if (k + 1 == f.size()) continue;  // already seeded with leading coef
        auto prod = poly::mul_truncated<Coef>(
            span<const Coef>(out.data(), out.size()),
            span<const Coef>(g_trunc.data(), g_trunc.size()), n);
        out = std::move(prod);
        out[0] = out[0] + f[k];
    }
    return out;
}

// Workspace-aware variant: caller supplies pre-sized scratch buffer to avoid
// repeated allocations across many compose calls (used by KL recursion base).
template <class Coef>
void compose_naive_horner_inplace(span<const Coef> f,
                                  span<const Coef> g,
                                  span<Coef> out,
                                  std::vector<Coef>& scratch) {
    detail::require_g0_zero(g);
    const std::size_t n = out.size();
    if (n == 0) return;

    for (auto& c : out) c = coef_zero<Coef>();
    if (f.empty()) return;

    if (scratch.size() < n) scratch.resize(n);
    auto g_trunc = poly::resized<Coef>(g, n);

    out[0] = f[f.size() - 1];
    for (std::size_t k = f.size() - 1; k-- > 0;) {
        // scratch := out * g_trunc truncated to n
        auto prod = poly::mul_truncated<Coef>(
            span<const Coef>(out.data(), n),
            span<const Coef>(g_trunc.data(), g_trunc.size()), n);
        for (std::size_t i = 0; i < n; ++i) out[i] = prod[i];
        out[0] = out[0] + f[k];
    }
}

}  // namespace pscomp::algo

#endif  // PSCOMP_COMPOSE_NAIVE_HPP
