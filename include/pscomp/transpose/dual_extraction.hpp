// Forward dual extraction: d_j = sum_k c_k * [x^k] g(x)^j, computed by
// bivariate Bostan-Mori in x on rev_n(c)(x) / (1 - y g(x)).

#ifndef PSCOMP_TRANSPOSE_DUAL_EXTRACTION_HPP
#define PSCOMP_TRANSPOSE_DUAL_EXTRACTION_HPP

#include <algorithm>
#include <cstddef>
#include <vector>

#include "pscomp/field/coef_traits.hpp"
#include "pscomp/poly/poly_mul.hpp"
#include "pscomp/span.hpp"
#include "pscomp/transpose/middle_product.hpp"

namespace pscomp::transpose {

template <class Coef>
using BiPoly = std::vector<std::vector<Coef>>;

namespace dual_detail {

template <class Coef>
std::size_t max_y(const BiPoly<Coef>& P) {
    std::size_t m = 0;
    for (const auto& r : P) m = std::max(m, r.size());
    return m;
}

template <class Coef>
void negate_odd_x_inplace(BiPoly<Coef>& P) {
    for (std::size_t i = 1; i < P.size(); i += 2) {
        for (auto& c : P[i]) c = coef_zero<Coef>() - c;
    }
}

template <class Coef>
BiPoly<Coef> take_x_parity(const BiPoly<Coef>& P, std::size_t parity) {
    if (P.size() <= parity) return BiPoly<Coef>{};
    const std::size_t out_n = (P.size() - parity + 1) / 2;
    BiPoly<Coef> R(out_n);
    for (std::size_t i = 0; i < out_n; ++i) {
        const std::size_t src = parity + 2 * i;
        if (src < P.size()) R[i] = P[src];
    }
    return R;
}

template <class Coef>
BiPoly<Coef> interleave_x_parity(const BiPoly<Coef>& P_acc,
                                 std::size_t parity,
                                 std::size_t out_n) {
    BiPoly<Coef> R(out_n);
    for (std::size_t i = 0; i < P_acc.size(); ++i) {
        const std::size_t dst = parity + 2 * i;
        if (dst < out_n) R[dst] = P_acc[i];
    }
    return R;
}

// Adjoint of bi_mul_x_truncated(P, B, x_cap, m_cap) with respect to P.
template <class Coef>
BiPoly<Coef> bi_middle_product_x(const BiPoly<Coef>& B,
                                 const BiPoly<Coef>& U,
                                 std::size_t x_cap,
                                 std::size_t m_cap) {
    BiPoly<Coef> R(x_cap);
    if (B.empty() || U.empty() || x_cap == 0 || m_cap == 0) return R;
    const std::size_t dy_b = max_y<Coef>(B);
    const std::size_t dy_u = max_y<Coef>(U);
    const std::size_t dy_out = std::min(m_cap, dy_u);
    if (dy_out == 0 || dy_b == 0) return R;
    for (auto& row : R) row.assign(dy_out, coef_zero<Coef>());

    const std::size_t dx_b = std::min(x_cap, B.size());
    const std::size_t dx_u = std::min(x_cap, U.size());

    std::vector<std::vector<Coef>> B_cols(dy_b, std::vector<Coef>(x_cap, coef_zero<Coef>()));
    for (std::size_t i = 0; i < dx_b; ++i) {
        for (std::size_t j = 0; j < B[i].size() && j < dy_b; ++j) B_cols[j][i] = B[i][j];
    }
    const std::size_t mp_x_len = 2 * x_cap - 1;
    std::vector<std::vector<Coef>> U_cols(dy_u, std::vector<Coef>(mp_x_len, coef_zero<Coef>()));
    for (std::size_t i = 0; i < dx_u; ++i) {
        for (std::size_t j = 0; j < U[i].size() && j < dy_u; ++j) U_cols[j][i] = U[i][j];
    }

    for (std::size_t jB = 0; jB < dy_b; ++jB) {
        for (std::size_t jU = jB; jU < dy_u; ++jU) {
            const std::size_t j_dst = jU - jB;
            if (j_dst >= dy_out) continue;
            auto col = middle_product<Coef>(
                span<const Coef>(B_cols[jB].data(), B_cols[jB].size()),
                span<const Coef>(U_cols[jU].data(), U_cols[jU].size()),
                x_cap);
            for (std::size_t i = 0; i < x_cap; ++i) {
                R[i][j_dst] = R[i][j_dst] + col[i];
            }
        }
    }
    return R;
}

// Bivariate convolution truncated to x_cap rows and m_cap y-columns,
// performed as one univariate convolution per (jA, jB) pair.
template <class Coef>
BiPoly<Coef> bi_mul_x_truncated(const BiPoly<Coef>& A,
                                 const BiPoly<Coef>& B,
                                 std::size_t x_cap,
                                 std::size_t m_cap) {
    BiPoly<Coef> R(x_cap);
    if (A.empty() || B.empty() || x_cap == 0 || m_cap == 0) return R;

    const std::size_t dy_a = max_y<Coef>(A);
    const std::size_t dy_b = max_y<Coef>(B);
    if (dy_a == 0 || dy_b == 0) return R;
    const std::size_t dy_out = std::min(m_cap, dy_a + dy_b - 1);
    for (auto& row : R) row.assign(dy_out, coef_zero<Coef>());

    const std::size_t dx_a = std::min(x_cap, A.size());
    const std::size_t dx_b = std::min(x_cap, B.size());

    std::vector<std::vector<Coef>> A_cols(dy_a, std::vector<Coef>(dx_a, coef_zero<Coef>()));
    for (std::size_t i = 0; i < dx_a; ++i) {
        for (std::size_t j = 0; j < A[i].size() && j < dy_a; ++j) A_cols[j][i] = A[i][j];
    }
    std::vector<std::vector<Coef>> B_cols(dy_b, std::vector<Coef>(dx_b, coef_zero<Coef>()));
    for (std::size_t i = 0; i < dx_b; ++i) {
        for (std::size_t j = 0; j < B[i].size() && j < dy_b; ++j) B_cols[j][i] = B[i][j];
    }

    for (std::size_t jA = 0; jA < dy_a; ++jA) {
        for (std::size_t jB = 0; jB < dy_b; ++jB) {
            const std::size_t j_dst = jA + jB;
            if (j_dst >= dy_out) continue;
            auto conv = poly::mul_truncated<Coef>(
                span<const Coef>(A_cols[jA].data(), A_cols[jA].size()),
                span<const Coef>(B_cols[jB].data(), B_cols[jB].size()),
                x_cap);
            for (std::size_t i = 0; i < x_cap; ++i) {
                R[i][j_dst] = R[i][j_dst] + conv[i];
            }
        }
    }
    return R;
}

template <class Coef>
struct SweepLevel {
    BiPoly<Coef> Q;
    BiPoly<Coef> Qm;
    std::size_t parity;
    std::size_t x_cap;
    std::size_t target_x;
};

// Per-level (Q, Qm, parity, x_cap) records derived from g, replayed both by
// extract_dual and by compose_kl_tellegen.
template <class Coef>
std::vector<SweepLevel<Coef>> sweep_forward_g(span<const Coef> g,
                                              std::size_t n,
                                              std::size_t m) {
    std::vector<SweepLevel<Coef>> out;
    if (n == 0 || m == 0) return out;

    BiPoly<Coef> Q(n);
    Q[0].assign(2, coef_zero<Coef>());
    Q[0][0] = coef_one<Coef>();
    {
        const std::size_t lim = std::min(n, g.size());
        for (std::size_t i = 1; i < lim; ++i) {
            Q[i].assign(2, coef_zero<Coef>());
            Q[i][1] = coef_zero<Coef>() - g[i];
        }
    }
    for (std::size_t i = 1; i < n; ++i) {
        if (Q[i].empty()) Q[i].assign(2, coef_zero<Coef>());
    }

    std::size_t target_x = n - 1;
    while (target_x > 0) {
        BiPoly<Coef> Qm = Q;
        negate_odd_x_inplace(Qm);
        const std::size_t parity = target_x & 1u;
        const std::size_t x_cap = target_x + 1;
        out.push_back(SweepLevel<Coef>{Q, Qm, parity, x_cap, target_x});
        BiPoly<Coef> V = bi_mul_x_truncated<Coef>(Q, Qm, x_cap, m);
        Q = take_x_parity<Coef>(V, 0);
        target_x = (target_x - parity) / 2;
    }
    return out;
}

}  // namespace dual_detail

template <class Coef>
std::vector<Coef> extract_dual(span<const Coef> c,
                               span<const Coef> g,
                               std::size_t n,
                               std::size_t m) {
    using dual_detail::bi_mul_x_truncated;
    using dual_detail::sweep_forward_g;
    using dual_detail::take_x_parity;
    std::vector<Coef> out(m, coef_zero<Coef>());
    if (n == 0 || m == 0) return out;

    BiPoly<Coef> P(n);
    {
        const std::size_t lim = std::min(n, c.size());
        for (std::size_t k = 0; k < lim; ++k) {
            P[n - 1 - k].assign(1, c[k]);
        }
        for (std::size_t i = 0; i < n; ++i) {
            if (P[i].empty()) P[i].assign(1, coef_zero<Coef>());
        }
    }

    auto levels = sweep_forward_g<Coef>(g, n, m);
    for (const auto& lvl : levels) {
        BiPoly<Coef> U = bi_mul_x_truncated<Coef>(P, lvl.Qm, lvl.x_cap, m);
        P = take_x_parity<Coef>(U, lvl.parity);
    }

    if (!P.empty() && !P[0].empty()) {
        for (std::size_t j = 0; j < m && j < P[0].size(); ++j) out[j] = P[0][j];
    }
    return out;
}

}  // namespace pscomp::transpose

#endif
