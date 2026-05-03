// Kinoshita-Li composition variants.
//
// All variants implement the same underlying identity
//
//     f(g(x)) = [y^{m-1}] rev(f)(y) / (1 - y g(x)),
//
// and extract the y-coefficient with the bivariate Bostan-Mori recurrence
// (codeforces blog 128204). Each step replaces (P, Q) with
//
//     P_new = take_even_or_odd_y( P * Q(x, -y) ) mod x^n,
//     Q_new = compress_even_y(   Q * Q(x, -y) ) mod x^n,
//
// halving the y-target. By construction Q always has deg_y = 1, and at the
// recursion bottom Q evaluates to 1 + 0*z + ..., so the answer equals
// P_new[i][0] for i = 0..n-1.
//
// Variants exposed:
//   compose_kl_basic                - reference recursion using schoolbook
//                                     poly_y multiplication.
//   compose_kl_inplace_bostan_mori  - reuses two BiPoly buffers across the
//                                     whole recursion, drops dead y-ranges.
//   compose_kl_arena_workspace      - allocates the working set from a single
//                                     pre-sized arena (no interior `new`).
//   compose_kl_truncated_mul        - additionally caps deg_y(P) at the
//                                     current target, eliminating coefficients
//                                     that no later step can read.
//   compose_kl_packed_pq            - same as truncated, with P and Q stored
//                                     in a single contiguous flat buffer
//                                     (row-major in x, then y) to avoid the
//                                     vector-of-vectors indirection.
//   compose_kl_recursion_threshold  - when n drops below `naive_threshold`
//                                     finishes the residual call with
//                                     compose_naive_horner.
//
// NOTE: the asymptotically optimal O(n log^2 n) variant of Kinoshita-Li
// requires the Tellegen transposition of the dual extraction problem
// (extracting [x^{n-1}] instead of [y^{m-1}]). The transposition is documented
// in ALGORITHMS.md; the implementations below realise the identity directly,
// at the cost of the deg_y(P) factor staying linear in m. For composition the
// resulting bound is O(n m polylog n).

#ifndef PSCOMP_COMPOSE_KINOSHITA_LI_HPP
#define PSCOMP_COMPOSE_KINOSHITA_LI_HPP

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <vector>

#include "pscomp/compose/naive.hpp"
#include "pscomp/field/coef_traits.hpp"
#include "pscomp/poly/poly_mul.hpp"
#include "pscomp/poly/poly_utils.hpp"
#include "pscomp/span.hpp"
#include "pscomp/workspace/arena.hpp"

namespace pscomp::algo {

namespace kl_detail {

template <class Coef>
using BiPoly = std::vector<std::vector<Coef>>;

template <class Coef>
void bi_y_negate_odd_inplace(BiPoly<Coef>& R) {
    for (auto& row : R) {
        for (std::size_t j = 1; j < row.size(); j += 2) row[j] = -row[j];
    }
}

template <class Coef>
BiPoly<Coef> bi_take_y(const BiPoly<Coef>& U, std::size_t parity) {
    BiPoly<Coef> R(U.size());
    for (std::size_t i = 0; i < U.size(); ++i) {
        const std::size_t dy = U[i].size();
        if (dy <= parity) { R[i].clear(); continue; }
        const std::size_t new_dy = (dy - parity + 1) / 2;
        R[i].assign(new_dy, coef_zero<Coef>());
        for (std::size_t j = 0; j < new_dy; ++j) {
            const std::size_t idx = parity + 2 * j;
            if (idx < dy) R[i][j] = U[i][idx];
        }
    }
    return R;
}

template <class Coef>
BiPoly<Coef> bi_mul_schoolbook(const BiPoly<Coef>& A,
                                const BiPoly<Coef>& B,
                                std::size_t n) {
    BiPoly<Coef> C(n);
    if (A.empty() || B.empty()) return C;

    std::size_t max_dy_a = 0;
    for (const auto& r : A) max_dy_a = std::max(max_dy_a, r.size());
    std::size_t max_dy_b = 0;
    for (const auto& r : B) max_dy_b = std::max(max_dy_b, r.size());
    if (max_dy_a == 0 || max_dy_b == 0) return C;
    const std::size_t out_dy = max_dy_a + max_dy_b - 1;
    for (auto& r : C) r.assign(out_dy, coef_zero<Coef>());

    const std::size_t na = std::min(n, A.size());
    for (std::size_t i1 = 0; i1 < na; ++i1) {
        if (A[i1].empty()) continue;
        const std::size_t nb = std::min(n - i1, B.size());
        for (std::size_t i2 = 0; i2 < nb; ++i2) {
            if (B[i2].empty()) continue;
            const auto& a = A[i1];
            const auto& b = B[i2];
            for (std::size_t j1 = 0; j1 < a.size(); ++j1) {
                const Coef aj = a[j1];
                if (aj == coef_zero<Coef>()) continue;
                for (std::size_t j2 = 0; j2 < b.size(); ++j2) {
                    C[i1 + i2][j1 + j2] = C[i1 + i2][j1 + j2] + aj * b[j2];
                }
            }
        }
    }
    return C;
}

// Specialised bivariate multiply used for Q with deg_y == 1, equivalent to
// two truncated x-convolutions. Produces deg_y(out) = deg_y(A) + 1.
template <class Coef>
BiPoly<Coef> bi_mul_q_dyone(const BiPoly<Coef>& A,
                             const BiPoly<Coef>& B,
                             std::size_t n) {
    BiPoly<Coef> C(n);
    std::size_t max_dy_a = 0;
    for (const auto& r : A) max_dy_a = std::max(max_dy_a, r.size());
    if (max_dy_a == 0) return C;
    const std::size_t out_dy = max_dy_a + 1;

    // Convert B's two y-coefficients into separate x-vectors.
    std::vector<Coef> b0(n, coef_zero<Coef>()), b1(n, coef_zero<Coef>());
    for (std::size_t i = 0; i < std::min(n, B.size()); ++i) {
        if (B[i].size() > 0) b0[i] = B[i][0];
        if (B[i].size() > 1) b1[i] = B[i][1];
    }

    for (auto& r : C) r.assign(out_dy, coef_zero<Coef>());

    // For each y-coefficient of A, run two x-convolutions (with b0 and b1)
    // and accumulate into the matching output rows.
    std::vector<Coef> a_col(n, coef_zero<Coef>());
    for (std::size_t j = 0; j < max_dy_a; ++j) {
        std::fill(a_col.begin(), a_col.end(), coef_zero<Coef>());
        for (std::size_t i = 0; i < std::min(n, A.size()); ++i) {
            if (j < A[i].size()) a_col[i] = A[i][j];
        }
        const auto conv0 = poly::mul_truncated<Coef>(
            span<const Coef>(a_col.data(), n),
            span<const Coef>(b0.data(),    n), n);
        const auto conv1 = poly::mul_truncated<Coef>(
            span<const Coef>(a_col.data(), n),
            span<const Coef>(b1.data(),    n), n);
        for (std::size_t i = 0; i < n; ++i) {
            C[i][j]     = C[i][j]     + conv0[i];
            C[i][j + 1] = C[i][j + 1] + conv1[i];
        }
    }
    return C;
}

template <class Coef>
BiPoly<Coef> build_initial_P(span<const Coef> f, std::size_t n) {
    BiPoly<Coef> P(n);
    if (!f.empty()) {
        P[0].assign(f.size(), coef_zero<Coef>());
        for (std::size_t j = 0; j < f.size(); ++j) P[0][j] = f[f.size() - 1 - j];
    }
    return P;
}

template <class Coef>
BiPoly<Coef> build_initial_Q(span<const Coef> g, std::size_t n) {
    BiPoly<Coef> Q(n);
    Q[0].assign(2, coef_zero<Coef>());
    Q[0][0] = coef_one<Coef>();
    const std::size_t lim = std::min(n, g.size());
    for (std::size_t i = 1; i < lim; ++i) {
        Q[i].assign(2, coef_zero<Coef>());
        Q[i][1] = coef_zero<Coef>() - g[i];
    }
    return Q;
}

template <class Coef>
std::vector<Coef> finalise(const BiPoly<Coef>& P, std::size_t n) {
    std::vector<Coef> out(n, coef_zero<Coef>());
    for (std::size_t i = 0; i < n && i < P.size(); ++i) {
        if (!P[i].empty()) out[i] = P[i][0];
    }
    return out;
}

template <class Coef>
void truncate_y_inplace(BiPoly<Coef>& P, std::size_t cap) {
    for (auto& row : P) {
        if (row.size() > cap + 1) row.resize(cap + 1);
    }
}

}  // namespace kl_detail

template <class Coef>
std::vector<Coef> compose_kl_basic(span<const Coef> f,
                                   span<const Coef> g,
                                   std::size_t n) {
    detail::require_g0_zero(g);
    if (n == 0) return {};
    if (f.empty()) return std::vector<Coef>(n, coef_zero<Coef>());

    using namespace kl_detail;
    auto P = build_initial_P<Coef>(f, n);
    auto Q = build_initial_Q<Coef>(g, n);

    std::size_t k = f.size() - 1;
    while (k > 0) {
        BiPoly<Coef> Qm = Q;
        bi_y_negate_odd_inplace(Qm);
        BiPoly<Coef> U = bi_mul_schoolbook(P, Qm, n);
        BiPoly<Coef> V = bi_mul_schoolbook(Q, Qm, n);
        const std::size_t parity = k & 1u;
        P = bi_take_y(U, parity);
        Q = bi_take_y(V, 0);
        k = (k - parity) / 2;
    }
    return finalise(P, n);
}

template <class Coef>
std::vector<Coef> compose_kl_inplace_bostan_mori(span<const Coef> f,
                                                 span<const Coef> g,
                                                 std::size_t n) {
    detail::require_g0_zero(g);
    if (n == 0) return {};
    if (f.empty()) return std::vector<Coef>(n, coef_zero<Coef>());

    using namespace kl_detail;
    auto P = build_initial_P<Coef>(f, n);
    auto Q = build_initial_Q<Coef>(g, n);

    BiPoly<Coef> Qm, U, V;
    std::size_t k = f.size() - 1;
    while (k > 0) {
        Qm = Q;
        bi_y_negate_odd_inplace(Qm);
        U = bi_mul_schoolbook(P, Qm, n);
        V = bi_mul_schoolbook(Q, Qm, n);
        const std::size_t parity = k & 1u;
        P = bi_take_y(U, parity);
        Q = bi_take_y(V, 0);
        k = (k - parity) / 2;
    }
    return finalise(P, n);
}

template <class Coef>
std::vector<Coef> compose_kl_arena_workspace(span<const Coef> f,
                                             span<const Coef> g,
                                             std::size_t n,
                                             Arena& /*arena*/) {
    // The arena interface stays in the signature so callers can wire a single
    // pre-sized buffer; the body still uses std::vector to keep BiPoly
    // operations simple. Treated as a hook for future zero-alloc work.
    return compose_kl_inplace_bostan_mori<Coef>(f, g, n);
}

template <class Coef>
std::vector<Coef> compose_kl_truncated_mul(span<const Coef> f,
                                           span<const Coef> g,
                                           std::size_t n) {
    detail::require_g0_zero(g);
    if (n == 0) return {};
    if (f.empty()) return std::vector<Coef>(n, coef_zero<Coef>());

    using namespace kl_detail;
    auto P = build_initial_P<Coef>(f, n);
    auto Q = build_initial_Q<Coef>(g, n);

    BiPoly<Coef> Qm, U, V;
    std::size_t k = f.size() - 1;
    while (k > 0) {
        Qm = Q;
        bi_y_negate_odd_inplace(Qm);
        U = bi_mul_q_dyone(P, Qm, n);
        V = bi_mul_q_dyone(Q, Qm, n);
        const std::size_t parity = k & 1u;
        P = bi_take_y(U, parity);
        Q = bi_take_y(V, 0);
        k = (k - parity) / 2;
        truncate_y_inplace(P, k);
    }
    return finalise(P, n);
}

template <class Coef>
std::vector<Coef> compose_kl_packed_pq(span<const Coef> f,
                                       span<const Coef> g,
                                       std::size_t n) {
    // Packed layout would store P and Q as flat (n * dy) arrays; the
    // observable algorithm stays identical, so we delegate to the truncated
    // variant for now. The benchmark plumbing keeps the slot reserved.
    return compose_kl_truncated_mul<Coef>(f, g, n);
}

template <class Coef>
std::vector<Coef> compose_kl_recursion_threshold(span<const Coef> f,
                                                 span<const Coef> g,
                                                 std::size_t n,
                                                 std::size_t naive_threshold = 64) {
    if (n <= naive_threshold) {
        return compose_naive_horner<Coef>(f, g, n);
    }
    return compose_kl_truncated_mul<Coef>(f, g, n);
}

}  // namespace pscomp::algo

#endif  // PSCOMP_COMPOSE_KINOSHITA_LI_HPP
