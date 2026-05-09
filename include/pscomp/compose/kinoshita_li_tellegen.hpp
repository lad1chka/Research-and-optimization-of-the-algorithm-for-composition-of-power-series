// Composition by transposing the dual extraction algorithm (Tellegen).

#ifndef PSCOMP_COMPOSE_KINOSHITA_LI_TELLEGEN_HPP
#define PSCOMP_COMPOSE_KINOSHITA_LI_TELLEGEN_HPP

#include <algorithm>
#include <cstddef>
#include <vector>

#include "pscomp/field/coef_traits.hpp"
#include "pscomp/span.hpp"
#include "pscomp/transpose/dual_extraction.hpp"

namespace pscomp::algo {

template <class Coef>
std::vector<Coef> compose_kl_tellegen(span<const Coef> f,
                                      span<const Coef> g,
                                      std::size_t n) {
    using transpose::dual_detail::bi_middle_product_x;
    using transpose::dual_detail::interleave_x_parity;
    using transpose::dual_detail::sweep_forward_g;
    using transpose::dual_detail::SweepLevel;
    using transpose::BiPoly;

    std::vector<Coef> out(n, coef_zero<Coef>());
    if (n == 0) return out;

    const std::size_t m = n;

    auto levels = sweep_forward_g<Coef>(g, n, m);

    BiPoly<Coef> P_acc(1);
    P_acc[0].assign(m, coef_zero<Coef>());
    {
        const std::size_t lim = std::min(m, f.size());
        for (std::size_t j = 0; j < lim; ++j) P_acc[0][j] = f[j];
    }

    for (auto it = levels.rbegin(); it != levels.rend(); ++it) {
        const auto& lvl = *it;
        BiPoly<Coef> U_t = interleave_x_parity<Coef>(P_acc, lvl.parity, lvl.x_cap);
        P_acc = bi_middle_product_x<Coef>(lvl.Qm, U_t, lvl.x_cap, m);
    }

    if (P_acc.size() < n) P_acc.resize(n);
    for (std::size_t k = 0; k < n; ++k) {
        const std::size_t src = n - 1 - k;
        if (src < P_acc.size() && !P_acc[src].empty()) {
            out[k] = P_acc[src][0];
        }
    }
    return out;
}

}  // namespace pscomp::algo

#endif
