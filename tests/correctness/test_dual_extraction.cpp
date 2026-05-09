// Tests extract_dual(c, g, n, m) against an O(n^2 m) oracle.

#include <gtest/gtest.h>

#include <cstdint>
#include <random>
#include <vector>

#include "pscomp/pscomp.hpp"
#include "pscomp/transpose/dual_extraction.hpp"
#include "test_helpers.hpp"

using pscomp::ModInt998;
using pscomp::span;
using pscomp::transpose::extract_dual;

namespace {

template <class Coef>
std::vector<Coef> mul_truncated(const std::vector<Coef>& a,
                                const std::vector<Coef>& b,
                                std::size_t n) {
    std::vector<Coef> out(n, pscomp::coef_zero<Coef>());
    for (std::size_t i = 0; i < std::min(n, a.size()); ++i) {
        if (a[i] == pscomp::coef_zero<Coef>()) continue;
        for (std::size_t j = 0; j + i < n && j < b.size(); ++j) {
            out[i + j] = out[i + j] + a[i] * b[j];
        }
    }
    return out;
}

template <class Coef>
std::vector<Coef> brute_dual(const std::vector<Coef>& c,
                             const std::vector<Coef>& g,
                             std::size_t n,
                             std::size_t m) {
    std::vector<Coef> out(m, pscomp::coef_zero<Coef>());
    if (n == 0 || m == 0) return out;

    std::vector<Coef> gpow(n, pscomp::coef_zero<Coef>());
    gpow[0] = pscomp::coef_one<Coef>();
    std::vector<Coef> g_padded(n, pscomp::coef_zero<Coef>());
    for (std::size_t i = 0; i < std::min(n, g.size()); ++i) g_padded[i] = g[i];

    for (std::size_t j = 0; j < m; ++j) {
        Coef acc = pscomp::coef_zero<Coef>();
        for (std::size_t k = 0; k < std::min(n, c.size()); ++k) {
            acc = acc + c[k] * gpow[k];
        }
        out[j] = acc;
        gpow = mul_truncated<Coef>(gpow, g_padded, n);
    }
    return out;
}

template <class Coef>
std::vector<Coef> random_mod(std::size_t n, std::mt19937& rng, bool zero_first = false) {
    std::uniform_int_distribution<std::uint32_t> dist(0u, pscomp::kPrime998 - 1u);
    std::vector<Coef> v(n);
    for (auto& x : v) x = Coef{dist(rng)};
    if (zero_first && !v.empty()) v[0] = Coef{};
    return v;
}

}  // namespace

TEST(ExtractDual, IdentityComposition) {
    const std::size_t n = 4, m = 4;
    std::vector<ModInt998> c = {ModInt998{3u}, ModInt998{1u},
                                 ModInt998{4u}, ModInt998{1u}};
    std::vector<ModInt998> g(n, pscomp::coef_zero<ModInt998>());
    g[1] = ModInt998{1u};

    auto got = extract_dual<ModInt998>(c, g, n, m);
    for (std::size_t j = 0; j < m; ++j) {
        EXPECT_EQ(got[j].to_uint(), c[j].to_uint())
            << "j=" << j;
    }
}

TEST(ExtractDual, ZeroG) {
    const std::size_t n = 4, m = 4;
    std::vector<ModInt998> c = {ModInt998{7u}, ModInt998{2u},
                                 ModInt998{5u}, ModInt998{9u}};
    std::vector<ModInt998> g(n, pscomp::coef_zero<ModInt998>());

    auto got = extract_dual<ModInt998>(c, g, n, m);
    EXPECT_EQ(got[0].to_uint(), c[0].to_uint());
    for (std::size_t j = 1; j < m; ++j) {
        EXPECT_EQ(got[j].to_uint(), 0u) << "j=" << j;
    }
}

TEST(ExtractDual, SmallRandomVsBrute) {
    std::mt19937 rng(pscomp_tests::kSeed ^ 0xAA11u);
    for (std::size_t n : {std::size_t{2}, std::size_t{3}, std::size_t{4},
                          std::size_t{6}, std::size_t{8}}) {
        for (std::size_t m : {std::size_t{1}, std::size_t{2},
                              std::size_t{3}, std::size_t{4}}) {
            auto c = random_mod<ModInt998>(n, rng);
            auto g = random_mod<ModInt998>(n, rng, true);

            auto ref = brute_dual<ModInt998>(c, g, n, m);
            auto got = extract_dual<ModInt998>(c, g, n, m);
            ASSERT_EQ(got.size(), m);
            for (std::size_t j = 0; j < m; ++j) {
                EXPECT_EQ(got[j].to_uint(), ref[j].to_uint())
                    << "n=" << n << " m=" << m << " j=" << j;
            }
        }
    }
}

TEST(ExtractDual, MediumRandomVsBrute) {
    std::mt19937 rng(pscomp_tests::kSeed ^ 0xAA22u);
    for (std::size_t n : {std::size_t{16}, std::size_t{24}, std::size_t{32}}) {
        for (std::size_t m : {std::size_t{4}, std::size_t{8}, std::size_t{16}}) {
            auto c = random_mod<ModInt998>(n, rng);
            auto g = random_mod<ModInt998>(n, rng, true);

            auto ref = brute_dual<ModInt998>(c, g, n, m);
            auto got = extract_dual<ModInt998>(c, g, n, m);
            ASSERT_EQ(got.size(), m);
            for (std::size_t j = 0; j < m; ++j) {
                EXPECT_EQ(got[j].to_uint(), ref[j].to_uint())
                    << "n=" << n << " m=" << m << " j=" << j;
            }
        }
    }
}
