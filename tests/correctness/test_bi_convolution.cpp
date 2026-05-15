#include <gtest/gtest.h>

#include <cstdint>
#include <random>
#include <vector>

#include "pscomp/pscomp.hpp"
#include "pscomp/transpose/bi_convolution.hpp"
#include "pscomp/transpose/dual_extraction.hpp"
#include "test_helpers.hpp"

using pscomp::ModInt998;
using pscomp::transpose::BiPoly;
using pscomp::transpose::bi_middle_product_x_y_ntt;
using pscomp::transpose::bi_mul_x_y_ntt;
using pscomp::transpose::dual_detail::bi_middle_product_x_naive;
using pscomp::transpose::dual_detail::bi_mul_x_truncated_naive;

namespace {

BiPoly<ModInt998> random_bipoly(std::size_t n_x, std::size_t n_y, std::mt19937& rng) {
    std::uniform_int_distribution<std::uint32_t> dist(0u, pscomp::kPrime998 - 1u);
    BiPoly<ModInt998> P(n_x);
    for (auto& row : P) {
        row.assign(n_y, ModInt998{});
        for (auto& c : row) c = ModInt998{dist(rng)};
    }
    return P;
}

void expect_eq(const BiPoly<ModInt998>& got, const BiPoly<ModInt998>& ref,
               const std::string& tag) {
    ASSERT_EQ(got.size(), ref.size()) << tag;
    for (std::size_t i = 0; i < got.size(); ++i) {
        ASSERT_EQ(got[i].size(), ref[i].size()) << tag << " row " << i;
        for (std::size_t j = 0; j < got[i].size(); ++j) {
            EXPECT_EQ(got[i][j].to_uint(), ref[i][j].to_uint())
                << tag << " (" << i << ", " << j << ")";
        }
    }
}

}  // namespace

TEST(BiConvolution, MulMatchesPairwise) {
    std::mt19937 rng(pscomp_tests::kSeed ^ 0xB011u);
    struct Case { std::size_t nx, ny_a, ny_b, x_cap, m_cap; };
    const std::vector<Case> cases = {
        {4, 2, 2,  4, 4},
        {8, 3, 1,  8, 4},
        {7, 5, 4,  7, 8},
        {16, 8, 8, 12, 8},
        {32, 4, 4, 32, 8},
        {64, 1, 1, 64, 1},
        {64, 16, 16, 64, 16},
    };
    for (auto c : cases) {
        auto A = random_bipoly(c.nx, c.ny_a, rng);
        auto B = random_bipoly(c.nx, c.ny_b, rng);
        auto ref = bi_mul_x_truncated_naive<ModInt998>(A, B, c.x_cap, c.m_cap);
        auto got = bi_mul_x_y_ntt<ModInt998>(A, B, c.x_cap, c.m_cap);
        expect_eq(got, ref,
                  "mul nx=" + std::to_string(c.nx) +
                  " ya=" + std::to_string(c.ny_a) +
                  " yb=" + std::to_string(c.ny_b) +
                  " xc=" + std::to_string(c.x_cap) +
                  " mc=" + std::to_string(c.m_cap));
    }
}

TEST(BiConvolution, MiddleMatchesPairwise) {
    std::mt19937 rng(pscomp_tests::kSeed ^ 0xB022u);
    struct Case { std::size_t b_nx, b_ny, u_nx, u_ny, x_cap, m_cap; };
    // U has length 2 * x_cap - 1 in x for the middle-product oracle.
    const std::vector<Case> cases = {
        {4, 2,  7, 2, 4, 4},
        {8, 1,  15, 4, 8, 4},
        {6, 3,  11, 5, 6, 8},
        {16, 4, 31, 8, 16, 8},
        {32, 8, 63, 16, 32, 16},
    };
    for (auto c : cases) {
        auto B = random_bipoly(c.b_nx, c.b_ny, rng);
        auto U = random_bipoly(c.u_nx, c.u_ny, rng);
        auto ref = bi_middle_product_x_naive<ModInt998>(B, U, c.x_cap, c.m_cap);
        auto got = bi_middle_product_x_y_ntt<ModInt998>(B, U, c.x_cap, c.m_cap);
        expect_eq(got, ref,
                  "mp bnx=" + std::to_string(c.b_nx) +
                  " bny=" + std::to_string(c.b_ny) +
                  " uny=" + std::to_string(c.u_ny) +
                  " xc=" + std::to_string(c.x_cap));
    }
}

TEST(BiConvolution, MulRandomized) {
    std::mt19937 rng(pscomp_tests::kSeed ^ 0xB033u);
    std::uniform_int_distribution<std::size_t> dim(1, 16);
    for (int t = 0; t < 30; ++t) {
        const std::size_t nx = dim(rng);
        const std::size_t ya = dim(rng);
        const std::size_t yb = dim(rng);
        const std::size_t xcap = std::max<std::size_t>(1, dim(rng));
        const std::size_t mcap = std::max<std::size_t>(1, dim(rng));
        auto A = random_bipoly(nx, ya, rng);
        auto B = random_bipoly(nx, yb, rng);
        auto ref = bi_mul_x_truncated_naive<ModInt998>(A, B, xcap, mcap);
        auto got = bi_mul_x_y_ntt<ModInt998>(A, B, xcap, mcap);
        expect_eq(got, ref, "mul rand t=" + std::to_string(t));
    }
}

TEST(BiConvolution, MiddleRandomized) {
    std::mt19937 rng(pscomp_tests::kSeed ^ 0xB044u);
    std::uniform_int_distribution<std::size_t> dim(1, 12);
    for (int t = 0; t < 30; ++t) {
        const std::size_t b_nx = dim(rng);
        const std::size_t b_ny = dim(rng);
        const std::size_t xcap = std::max<std::size_t>(b_nx, dim(rng));
        const std::size_t u_nx = 2 * xcap - 1;
        const std::size_t u_ny = std::max<std::size_t>(b_ny, dim(rng));
        const std::size_t mcap = std::max<std::size_t>(1, dim(rng));
        auto B = random_bipoly(b_nx, b_ny, rng);
        auto U = random_bipoly(u_nx, u_ny, rng);
        auto ref = bi_middle_product_x_naive<ModInt998>(B, U, xcap, mcap);
        auto got = bi_middle_product_x_y_ntt<ModInt998>(B, U, xcap, mcap);
        expect_eq(got, ref, "mp rand t=" + std::to_string(t));
    }
}
