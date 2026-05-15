// Joint 2D NTT/FFT primitives for bivariate convolution and middle product

#ifndef PSCOMP_TRANSPOSE_BI_CONVOLUTION_HPP
#define PSCOMP_TRANSPOSE_BI_CONVOLUTION_HPP

#include <algorithm>
#include <complex>
#include <cstddef>
#include <type_traits>
#include <vector>

#include "pscomp/field/coef_traits.hpp"
#include "pscomp/field/mod_int.hpp"
#include "pscomp/poly/poly_utils.hpp"
#include "pscomp/span.hpp"
#include "pscomp/transform/fft.hpp"
#include "pscomp/transform/ntt.hpp"
#include "pscomp/transform/ntt_basic.hpp"

namespace pscomp::transpose {

template <class Coef>
using BiPoly = std::vector<std::vector<Coef>>;

namespace bi_detail {

// Element type used in the transform domain.
template <class Coef>
using transform_elem_t = std::conditional_t<
    std::is_same_v<Coef, ModInt998>     , ModInt998,
    std::conditional_t<
    std::is_same_v<Coef, ModInt998Plain>, ModInt998Plain,
    std::complex<Coef>>>;

template <class Coef>
void transform_1d(span<transform_elem_t<Coef>> a, bool inverse) {
    if constexpr (std::is_same_v<Coef, ModInt998>) {
        ntt::transform(a, inverse);
    } else if constexpr (std::is_same_v<Coef, ModInt998Plain>) {
        ntt_basic::transform(a, inverse);
    } else {
        fft::transform<Coef>(a, inverse);
    }
}

template <class Coef>
transform_elem_t<Coef> to_transform(Coef x) {
    if constexpr (std::is_same_v<Coef, ModInt998> ||
                  std::is_same_v<Coef, ModInt998Plain>) {
        return x;
    } else {
        return std::complex<Coef>(x, 0);
    }
}

template <class Coef>
Coef from_transform(transform_elem_t<Coef> x) {
    if constexpr (std::is_same_v<Coef, ModInt998> ||
                  std::is_same_v<Coef, ModInt998Plain>) {
        return x;
    } else {
        return x.real();
    }
}

template <class Coef>
std::vector<transform_elem_t<Coef>>& scratch_buffer(int slot) {
    using E = transform_elem_t<Coef>;
    thread_local std::vector<E> bufs[4];
    return bufs[slot];
}

template <class Coef>
std::vector<transform_elem_t<Coef>>& take_scratch(int slot, std::size_t total) {
    auto& buf = scratch_buffer<Coef>(slot);
    if (buf.size() < total) buf.assign(total, transform_elem_t<Coef>{});
    else std::fill(buf.begin(), buf.begin() + total, transform_elem_t<Coef>{});
    return buf;
}

template <class Coef>
void pack_bipoly_into(std::vector<transform_elem_t<Coef>>& buf,
                      const BiPoly<Coef>& A, std::size_t nx_used,
                      std::size_t ny_used, std::size_t Lx, std::size_t Ly) {
    const std::size_t nx = std::min({nx_used, Lx, A.size()});
    for (std::size_t i = 0; i < nx; ++i) {
        const auto& row = A[i];
        const std::size_t ny = std::min({ny_used, Ly, row.size()});
        auto* dst = buf.data() + i * Ly;
        for (std::size_t j = 0; j < ny; ++j) dst[j] = to_transform<Coef>(row[j]);
    }
}

template <class Coef>
void transform_2d(std::vector<transform_elem_t<Coef>>& buf,
                  std::size_t Lx, std::size_t Ly, bool inverse) {
    using E = transform_elem_t<Coef>;
    if (Ly > 1) {
        for (std::size_t i = 0; i < Lx; ++i) {
            transform_1d<Coef>(span<E>(buf.data() + i * Ly, Ly), inverse);
        }
    }
    if (Lx > 1) {
        thread_local std::vector<E> col;
        col.assign(Lx, E{});
        for (std::size_t j = 0; j < Ly; ++j) {
            for (std::size_t i = 0; i < Lx; ++i) col[i] = buf[i * Ly + j];
            transform_1d<Coef>(span<E>(col.data(), Lx), inverse);
            for (std::size_t i = 0; i < Lx; ++i) buf[i * Ly + j] = col[i];
        }
    }
}

template <class Coef>
void pointwise_mul(std::vector<transform_elem_t<Coef>>& a,
                   const std::vector<transform_elem_t<Coef>>& b) {
    const std::size_t n = a.size();
    for (std::size_t i = 0; i < n; ++i) a[i] = a[i] * b[i];
}

}  // namespace bi_detail

template <class Coef>
BiPoly<Coef> bi_mul_x_y_ntt(const BiPoly<Coef>& A,
                            const BiPoly<Coef>& B,
                            std::size_t x_cap,
                            std::size_t m_cap) {
    BiPoly<Coef> R(x_cap);
    if (A.empty() || B.empty() || x_cap == 0 || m_cap == 0) return R;

    std::size_t dy_a = 0, dy_b = 0;
    for (const auto& row : A) dy_a = std::max(dy_a, row.size());
    for (const auto& row : B) dy_b = std::max(dy_b, row.size());
    if (dy_a == 0 || dy_b == 0) return R;

    const std::size_t dx_a = std::min(x_cap, A.size());
    const std::size_t dx_b = std::min(x_cap, B.size());
    const std::size_t dy_out = std::min(m_cap, dy_a + dy_b - 1);
    for (auto& row : R) row.assign(dy_out, coef_zero<Coef>());

    const std::size_t conv_x = dx_a + dx_b - 1;
    const std::size_t Lx = poly::round_up_pow2(std::max<std::size_t>(2, conv_x));
    const std::size_t Ly = poly::round_up_pow2(std::max<std::size_t>(2, dy_a + dy_b));

    const std::size_t total = Lx * Ly;
    auto& fa = bi_detail::take_scratch<Coef>(0, total);
    auto& fb = bi_detail::take_scratch<Coef>(1, total);
    bi_detail::pack_bipoly_into<Coef>(fa, A, dx_a, dy_a, Lx, Ly);
    bi_detail::pack_bipoly_into<Coef>(fb, B, dx_b, dy_b, Lx, Ly);
    bi_detail::transform_2d<Coef>(fa, Lx, Ly, false);
    bi_detail::transform_2d<Coef>(fb, Lx, Ly, false);
    bi_detail::pointwise_mul<Coef>(fa, fb);
    bi_detail::transform_2d<Coef>(fa, Lx, Ly, true);

    const std::size_t out_x = std::min(x_cap, conv_x);
    for (std::size_t i = 0; i < out_x; ++i) {
        const auto* src = fa.data() + i * Ly;
        for (std::size_t j = 0; j < dy_out; ++j) {
            R[i][j] = bi_detail::from_transform<Coef>(src[j]);
        }
    }
    return R;
}

template <class Coef>
BiPoly<Coef> bi_middle_product_x_y_ntt(const BiPoly<Coef>& B,
                                       const BiPoly<Coef>& U,
                                       std::size_t x_cap,
                                       std::size_t m_cap) {
    BiPoly<Coef> R(x_cap);
    if (B.empty() || U.empty() || x_cap == 0 || m_cap == 0) return R;

    std::size_t dy_b = 0, dy_u = 0;
    for (const auto& row : B) dy_b = std::max(dy_b, row.size());
    for (const auto& row : U) dy_u = std::max(dy_u, row.size());
    const std::size_t dy_out = std::min(m_cap, dy_u);
    if (dy_b == 0 || dy_out == 0) return R;
    for (auto& row : R) row.assign(dy_out, coef_zero<Coef>());

    const std::size_t dx_b_used = std::min(x_cap, B.size());
    const std::size_t dx_u_used = std::min(x_cap, U.size());

    const std::size_t conv_x = x_cap + (2 * x_cap - 1) - 1;
    const std::size_t Lx = poly::round_up_pow2(std::max<std::size_t>(2, conv_x));
    const std::size_t Ly = poly::round_up_pow2(std::max<std::size_t>(2, dy_b + dy_u));

    const std::size_t total = Lx * Ly;
    auto& fb = bi_detail::take_scratch<Coef>(0, total);
    auto& fu = bi_detail::take_scratch<Coef>(1, total);
    for (std::size_t i_src = 0; i_src < dx_b_used; ++i_src) {
        const std::size_t i_dst = (x_cap - 1) - i_src;
        const auto& row = B[i_src];
        const std::size_t ny = std::min(dy_b, row.size());
        auto* dst = fb.data() + i_dst * Ly;
        for (std::size_t j = 0; j < ny; ++j) {
            dst[dy_b - 1 - j] = bi_detail::to_transform<Coef>(row[j]);
        }
    }
    for (std::size_t i = 0; i < dx_u_used; ++i) {
        const auto& row = U[i];
        const std::size_t ny = std::min(dy_u, row.size());
        auto* dst = fu.data() + i * Ly;
        for (std::size_t j = 0; j < ny; ++j) {
            dst[j] = bi_detail::to_transform<Coef>(row[j]);
        }
    }

    bi_detail::transform_2d<Coef>(fb, Lx, Ly, false);
    bi_detail::transform_2d<Coef>(fu, Lx, Ly, false);
    bi_detail::pointwise_mul<Coef>(fb, fu);
    bi_detail::transform_2d<Coef>(fb, Lx, Ly, true);

    const std::size_t i_base = x_cap - 1;
    const std::size_t j_base = dy_b - 1;
    for (std::size_t i = 0; i < x_cap; ++i) {
        const auto* src = fb.data() + (i_base + i) * Ly;
        for (std::size_t j = 0; j < dy_out; ++j) {
            R[i][j] = bi_detail::from_transform<Coef>(src[j_base + j]);
        }
    }
    return R;
}

template <class Coef>
BiPoly<Coef> bi_mul_y_ntt(const BiPoly<Coef>& A,
                          const BiPoly<Coef>& B,
                          std::size_t n_x,
                          std::size_t m_cap) {
    BiPoly<Coef> R(n_x);
    if (A.empty() || B.empty() || n_x == 0 || m_cap == 0) return R;

    std::size_t dy_a = 0, dy_b = 0;
    for (const auto& row : A) dy_a = std::max(dy_a, row.size());
    for (const auto& row : B) dy_b = std::max(dy_b, row.size());
    if (dy_a == 0 || dy_b == 0) return R;

    const std::size_t dx_a = std::min(n_x, A.size());
    const std::size_t dx_b = std::min(n_x, B.size());
    const std::size_t conv_y = dy_a + dy_b - 1;
    const std::size_t dy_out = std::min(m_cap, conv_y);
    for (auto& row : R) row.assign(dy_out, coef_zero<Coef>());

    const std::size_t conv_x = dx_a + dx_b - 1;
    const std::size_t Lx = poly::round_up_pow2(std::max<std::size_t>(2, conv_x));
    const std::size_t Ly = poly::round_up_pow2(std::max<std::size_t>(2, conv_y));

    const std::size_t total = Lx * Ly;
    auto& fa = bi_detail::take_scratch<Coef>(0, total);
    auto& fb = bi_detail::take_scratch<Coef>(1, total);
    bi_detail::pack_bipoly_into<Coef>(fa, A, dx_a, dy_a, Lx, Ly);
    bi_detail::pack_bipoly_into<Coef>(fb, B, dx_b, dy_b, Lx, Ly);
    bi_detail::transform_2d<Coef>(fa, Lx, Ly, false);
    bi_detail::transform_2d<Coef>(fb, Lx, Ly, false);
    bi_detail::pointwise_mul<Coef>(fa, fb);
    bi_detail::transform_2d<Coef>(fa, Lx, Ly, true);

    const std::size_t out_x = std::min(n_x, conv_x);
    for (std::size_t i = 0; i < out_x; ++i) {
        const auto* src = fa.data() + i * Ly;
        for (std::size_t j = 0; j < dy_out; ++j) {
            R[i][j] = bi_detail::from_transform<Coef>(src[j]);
        }
    }
    return R;
}

template <class Coef>
BiPoly<Coef> bi_middle_product_y_ntt(const BiPoly<Coef>& B,
                                     const BiPoly<Coef>& U,
                                     std::size_t n_x,
                                     std::size_t m_cap) {
    BiPoly<Coef> R(n_x);
    if (B.empty() || U.empty() || n_x == 0 || m_cap == 0) return R;

    std::size_t dy_b_full = 0, dy_u_full = 0;
    for (const auto& row : B) dy_b_full = std::max(dy_b_full, row.size());
    for (const auto& row : U) dy_u_full = std::max(dy_u_full, row.size());
    if (dy_b_full == 0 || dy_u_full == 0) return R;

    const std::size_t dy_b_used = std::min(m_cap, dy_b_full);
    const std::size_t dy_u_used = std::min(m_cap, dy_u_full);

    const std::size_t dx_b = std::min(n_x, B.size());
    const std::size_t dx_u = std::min(n_x, U.size());
    const std::size_t dx_out = std::min(n_x, dx_u);
    if (dx_out == 0) return R;
    for (std::size_t i = 0; i < dx_out; ++i) R[i].assign(m_cap, coef_zero<Coef>());

    const std::size_t conv_x = dx_b + dx_u - 1;
    const std::size_t conv_y = m_cap + (2 * m_cap - 1) - 1;
    const std::size_t Lx = poly::round_up_pow2(std::max<std::size_t>(2, conv_x));
    const std::size_t Ly = poly::round_up_pow2(std::max<std::size_t>(2, conv_y));

    const std::size_t total = Lx * Ly;
    auto& fb = bi_detail::take_scratch<Coef>(0, total);
    auto& fu = bi_detail::take_scratch<Coef>(1, total);
    for (std::size_t i = 0; i < dx_b; ++i) {
        const auto& row = B[i];
        const std::size_t ny = std::min(dy_b_used, row.size());
        auto* dst = fb.data() + i * Ly;
        for (std::size_t j = 0; j < ny; ++j) {
            dst[m_cap - 1 - j] = bi_detail::to_transform<Coef>(row[j]);
        }
    }
    for (std::size_t i = 0; i < dx_u; ++i) {
        const auto& row = U[i];
        const std::size_t ny = std::min(dy_u_used, row.size());
        auto* dst = fu.data() + i * Ly;
        for (std::size_t j = 0; j < ny; ++j) {
            dst[j] = bi_detail::to_transform<Coef>(row[j]);
        }
    }

    bi_detail::transform_2d<Coef>(fb, Lx, Ly, false);
    bi_detail::transform_2d<Coef>(fu, Lx, Ly, false);
    bi_detail::pointwise_mul<Coef>(fb, fu);
    bi_detail::transform_2d<Coef>(fb, Lx, Ly, true);

    const std::size_t j_base = m_cap - 1;
    for (std::size_t i = 0; i < dx_out; ++i) {
        const auto* src = fb.data() + i * Ly;
        for (std::size_t j = 0; j < m_cap; ++j) {
            R[i][j] = bi_detail::from_transform<Coef>(src[j_base + j]);
        }
    }
    return R;
}

}  // namespace pscomp::transpose

#endif
