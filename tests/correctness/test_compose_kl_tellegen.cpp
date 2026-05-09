// Bit-for-bit equality of compose_kl_tellegen vs compose_kl_basic on ModInt998.

#include <gtest/gtest.h>

#include <cstdint>
#include <random>
#include <vector>

#include "pscomp/pscomp.hpp"
#include "pscomp/compose/kinoshita_li.hpp"
#include "pscomp/compose/kinoshita_li_tellegen.hpp"
#include "test_helpers.hpp"

using pscomp::ModInt998;
using pscomp::span;

namespace {

template <class Coef>
std::vector<Coef> random_mod(std::size_t n, std::mt19937& rng,
                             bool zero_first = false) {
    std::uniform_int_distribution<std::uint32_t> dist(0u, pscomp::kPrime998 - 1u);
    std::vector<Coef> v(n);
    for (auto& x : v) x = Coef{dist(rng)};
    if (zero_first && !v.empty()) v[0] = Coef{};
    return v;
}

void expect_eq_vec(const std::vector<ModInt998>& got,
                   const std::vector<ModInt998>& ref,
                   const std::string& tag) {
    ASSERT_EQ(got.size(), ref.size()) << tag;
    for (std::size_t i = 0; i < got.size(); ++i) {
        EXPECT_EQ(got[i].to_uint(), ref[i].to_uint())
            << tag << " idx=" << i;
    }
}

}  // namespace

TEST(ComposeKLTellegen, IdentityG) {
    const std::size_t n = 8;
    std::vector<ModInt998> f = {ModInt998{3u}, ModInt998{1u}, ModInt998{4u},
                                 ModInt998{1u}, ModInt998{5u}, ModInt998{9u},
                                 ModInt998{2u}, ModInt998{6u}};
    std::vector<ModInt998> g(n, pscomp::coef_zero<ModInt998>());
    g[1] = ModInt998{1u};

    auto ref = pscomp::algo::compose_kl_basic<ModInt998>(f, g, n);
    auto got = pscomp::algo::compose_kl_tellegen<ModInt998>(f, g, n);
    expect_eq_vec(got, ref, "g=x");
}

TEST(ComposeKLTellegen, FIsE0) {
    std::mt19937 rng(pscomp_tests::kSeed ^ 0xCC11u);
    const std::size_t n = 8;
    std::vector<ModInt998> f(n, pscomp::coef_zero<ModInt998>());
    f[0] = ModInt998{42u};
    auto g = random_mod<ModInt998>(n, rng, true);

    auto ref = pscomp::algo::compose_kl_basic<ModInt998>(f, g, n);
    auto got = pscomp::algo::compose_kl_tellegen<ModInt998>(f, g, n);
    expect_eq_vec(got, ref, "f=e_0");
}

TEST(ComposeKLTellegen, FIsE1) {
    std::mt19937 rng(pscomp_tests::kSeed ^ 0xCC22u);
    const std::size_t n = 8;
    std::vector<ModInt998> f(n, pscomp::coef_zero<ModInt998>());
    f[1] = ModInt998{1u};
    auto g = random_mod<ModInt998>(n, rng, true);

    auto ref = pscomp::algo::compose_kl_basic<ModInt998>(f, g, n);
    auto got = pscomp::algo::compose_kl_tellegen<ModInt998>(f, g, n);
    expect_eq_vec(got, ref, "f=e_1");
}

TEST(ComposeKLTellegen, SmallRandomVsBasic) {
    std::mt19937 rng(pscomp_tests::kSeed ^ 0xCC33u);
    for (std::size_t n : {std::size_t{2}, std::size_t{3}, std::size_t{4},
                          std::size_t{6}, std::size_t{8}, std::size_t{12},
                          std::size_t{16}}) {
        auto f = random_mod<ModInt998>(n, rng);
        auto g = random_mod<ModInt998>(n, rng, true);
        auto ref = pscomp::algo::compose_kl_basic<ModInt998>(f, g, n);
        auto got = pscomp::algo::compose_kl_tellegen<ModInt998>(f, g, n);
        expect_eq_vec(got, ref, "small n=" + std::to_string(n));
    }
}

TEST(ComposeKLTellegen, MediumRandomVsBasic) {
    std::mt19937 rng(pscomp_tests::kSeed ^ 0xCC44u);
    for (std::size_t n : {std::size_t{24}, std::size_t{32}, std::size_t{48},
                          std::size_t{64}}) {
        auto f = random_mod<ModInt998>(n, rng);
        auto g = random_mod<ModInt998>(n, rng, true);
        auto ref = pscomp::algo::compose_kl_basic<ModInt998>(f, g, n);
        auto got = pscomp::algo::compose_kl_tellegen<ModInt998>(f, g, n);
        expect_eq_vec(got, ref, "medium n=" + std::to_string(n));
    }
}
