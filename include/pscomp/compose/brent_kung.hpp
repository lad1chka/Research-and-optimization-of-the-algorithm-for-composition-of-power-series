// Brent-Kung composition (1978): O(sqrt(n) * M(n)) multiplications.

#ifndef PSCOMP_COMPOSE_BRENT_KUNG_HPP
#define PSCOMP_COMPOSE_BRENT_KUNG_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "pscomp/compose/naive.hpp"
#include "pscomp/field/coef_traits.hpp"
#include "pscomp/poly/poly_mul.hpp"
#include "pscomp/poly/poly_utils.hpp"
#include "pscomp/span.hpp"

namespace pscomp::algo {

namespace bk_detail {

inline std::size_t default_block_size(std::size_t n) {
    if (n <= 1) return 1;
    auto m = static_cast<std::size_t>(std::ceil(std::sqrt(static_cast<double>(n))));
    return std::max<std::size_t>(m, 1);
}

inline std::size_t tuned_block_size(std::size_t n) {
    return poly::round_up_pow2(default_block_size(n));
}

}  // namespace bk_detail

template <class Coef>
std::vector<Coef> compose_brent_kung_basic(span<const Coef> f,
                                           span<const Coef> g,
                                           std::size_t n) {
    detail::require_g0_zero(g);
    if (n == 0) return {};
    if (f.empty()) return std::vector<Coef>(n, coef_zero<Coef>());

    const std::size_t m      = bk_detail::default_block_size(n);
    const std::size_t blocks = (f.size() + m - 1) / m;

    std::vector<std::vector<Coef>> g_pow(m + 1);
    g_pow[0].assign(n, coef_zero<Coef>());
    g_pow[0][0] = coef_one<Coef>();
    g_pow[1] = poly::resized<Coef>(g, n);
    for (std::size_t k = 2; k <= m; ++k) {
        g_pow[k] = poly::mul_truncated<Coef>(
            span<const Coef>(g_pow[k - 1].data(), n),
            span<const Coef>(g_pow[1].data(),     n), n);
    }

    std::vector<std::vector<Coef>> block_value(blocks);
    for (std::size_t k = 0; k < blocks; ++k) {
        block_value[k].assign(n, coef_zero<Coef>());
        for (std::size_t j = 0; j < m; ++j) {
            const std::size_t idx = k * m + j;
            if (idx >= f.size()) break;
            const Coef coef = f[idx];
            if (coef == coef_zero<Coef>()) continue;
            const auto& gp = g_pow[j];
            for (std::size_t i = 0; i < n; ++i) {
                block_value[k][i] = block_value[k][i] + coef * gp[i];
            }
        }
    }

    std::vector<Coef> out = std::move(block_value[blocks - 1]);
    for (std::size_t k = blocks - 1; k-- > 0;) {
        out = poly::mul_truncated<Coef>(
            span<const Coef>(out.data(), n),
            span<const Coef>(g_pow[m].data(), n), n);
        for (std::size_t i = 0; i < n; ++i) out[i] = out[i] + block_value[k][i];
    }
    return out;
}

template <class Coef>
std::vector<Coef> compose_brent_kung_opt(span<const Coef> f,
                                         span<const Coef> g,
                                         std::size_t n) {
    detail::require_g0_zero(g);
    if (n == 0) return {};
    if (f.empty()) return std::vector<Coef>(n, coef_zero<Coef>());

    const std::size_t m      = bk_detail::default_block_size(n);
    const std::size_t blocks = (f.size() + m - 1) / m;

    std::vector<std::vector<Coef>> g_pow(m + 1);
    g_pow[0].assign(n, coef_zero<Coef>());
    g_pow[0][0] = coef_one<Coef>();
    g_pow[1] = poly::resized<Coef>(g, n);
    for (std::size_t k = 2; k <= m; ++k) {
        g_pow[k] = poly::mul_truncated<Coef>(
            span<const Coef>(g_pow[k - 1].data(), n),
            span<const Coef>(g_pow[1].data(),     n), n);
    }

    std::vector<Coef> out(n, coef_zero<Coef>());
    std::vector<Coef> block_buf(n, coef_zero<Coef>());
    for (std::size_t k = blocks; k-- > 0;) {
        std::fill(block_buf.begin(), block_buf.end(), coef_zero<Coef>());
        for (std::size_t j = 0; j < m; ++j) {
            const std::size_t idx = k * m + j;
            if (idx >= f.size()) break;
            const Coef coef = f[idx];
            if (coef == coef_zero<Coef>()) continue;
            const auto& gp = g_pow[j];
            for (std::size_t i = 0; i < n; ++i) {
                block_buf[i] = block_buf[i] + coef * gp[i];
            }
        }

        if (k + 1 == blocks) {
            out = block_buf;
        } else {
            out = poly::mul_truncated<Coef>(
                span<const Coef>(out.data(), n),
                span<const Coef>(g_pow[m].data(), n), n);
            for (std::size_t i = 0; i < n; ++i) out[i] = out[i] + block_buf[i];
        }
    }
    return out;
}

template <class Coef>
std::vector<Coef> compose_brent_kung_streaming(span<const Coef> f,
                                               span<const Coef> g,
                                               std::size_t n) {
    detail::require_g0_zero(g);
    if (n == 0) return {};
    if (f.empty()) return std::vector<Coef>(n, coef_zero<Coef>());

    const std::size_t m      = bk_detail::default_block_size(n);
    const std::size_t blocks = (f.size() + m - 1) / m;

    std::vector<std::vector<Coef>> block_value(blocks);
    for (auto& bv : block_value) bv.assign(n, coef_zero<Coef>());

    std::vector<Coef> g_j(n, coef_zero<Coef>());
    g_j[0] = coef_one<Coef>();
    std::vector<Coef> g_trunc = poly::resized<Coef>(g, n);

    std::vector<Coef> g_m;
    for (std::size_t j = 0; j < m; ++j) {
        for (std::size_t k = 0; k < blocks; ++k) {
            const std::size_t idx = k * m + j;
            if (idx >= f.size()) break;
            const Coef coef = f[idx];
            if (coef == coef_zero<Coef>()) continue;
            for (std::size_t i = 0; i < n; ++i) {
                block_value[k][i] = block_value[k][i] + coef * g_j[i];
            }
        }
        if (j + 1 == m) g_m = g_j;
        if (j + 1 < m) {
            g_j = poly::mul_truncated<Coef>(
                span<const Coef>(g_j.data(),     n),
                span<const Coef>(g_trunc.data(), n), n);
        }
    }
    g_m = poly::mul_truncated<Coef>(
        span<const Coef>(g_j.data(),     n),
        span<const Coef>(g_trunc.data(), n), n);

    std::vector<Coef> out = std::move(block_value[blocks - 1]);
    for (std::size_t k = blocks - 1; k-- > 0;) {
        out = poly::mul_truncated<Coef>(
            span<const Coef>(out.data(),  n),
            span<const Coef>(g_m.data(),  n), n);
        for (std::size_t i = 0; i < n; ++i) out[i] = out[i] + block_value[k][i];
    }
    return out;
}

template <class Coef>
std::vector<Coef> compose_brent_kung_tuned_m(span<const Coef> f,
                                             span<const Coef> g,
                                             std::size_t n) {
    detail::require_g0_zero(g);
    if (n == 0) return {};
    if (f.empty()) return std::vector<Coef>(n, coef_zero<Coef>());

    const std::size_t m      = bk_detail::tuned_block_size(n);
    const std::size_t blocks = (f.size() + m - 1) / m;

    std::vector<std::vector<Coef>> g_pow(m + 1);
    g_pow[0].assign(n, coef_zero<Coef>());
    g_pow[0][0] = coef_one<Coef>();
    g_pow[1] = poly::resized<Coef>(g, n);
    for (std::size_t k = 2; k <= m; ++k) {
        g_pow[k] = poly::mul_truncated<Coef>(
            span<const Coef>(g_pow[k - 1].data(), n),
            span<const Coef>(g_pow[1].data(),     n), n);
    }

    std::vector<Coef> out(n, coef_zero<Coef>());
    std::vector<Coef> block_buf(n, coef_zero<Coef>());
    for (std::size_t k = blocks; k-- > 0;) {
        std::fill(block_buf.begin(), block_buf.end(), coef_zero<Coef>());
        for (std::size_t j = 0; j < m; ++j) {
            const std::size_t idx = k * m + j;
            if (idx >= f.size()) break;
            const Coef coef = f[idx];
            if (coef == coef_zero<Coef>()) continue;
            const auto& gp = g_pow[j];
            for (std::size_t i = 0; i < n; ++i) {
                block_buf[i] = block_buf[i] + coef * gp[i];
            }
        }
        if (k + 1 == blocks) {
            out = block_buf;
        } else {
            out = poly::mul_truncated<Coef>(
                span<const Coef>(out.data(), n),
                span<const Coef>(g_pow[m].data(), n), n);
            for (std::size_t i = 0; i < n; ++i) out[i] = out[i] + block_buf[i];
        }
    }
    return out;
}

}  // namespace pscomp::algo

#endif
