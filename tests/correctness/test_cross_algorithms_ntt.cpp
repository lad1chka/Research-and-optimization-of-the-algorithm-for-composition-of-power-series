// Cross-check every compose variant against compose_naive_horner on NTT inputs.

#include <gtest/gtest.h>

#include "test_helpers.hpp"

using pscomp::ModInt998;
using namespace pscomp_tests;

namespace {

void compare_against_horner(std::size_t n) {
    std::mt19937 rng(kSeed ^ static_cast<std::uint32_t>(n));
    auto f = random_ntt(n, rng);
    auto g = random_ntt(n, rng, true);

    auto reference = pscomp::algo::compose_naive_horner<ModInt998>(f, g, n);
    for (auto& entry : all_entries<ModInt998>(n <= 32)) {
        auto out = entry.fn(f, g, n);
        ASSERT_EQ(out.size(), reference.size()) << entry.name << " n=" << n;
        for (std::size_t i = 0; i < n; ++i) {
            EXPECT_EQ(out[i], reference[i])
                << entry.name << " coef[" << i << "] n=" << n;
        }
    }
}

}  // namespace

TEST(CrossAlgorithmNTT, SmallN16)  { compare_against_horner(16);  }
TEST(CrossAlgorithmNTT, MediumN64) { compare_against_horner(64);  }
TEST(CrossAlgorithmNTT, N128)      { compare_against_horner(128); }
TEST(CrossAlgorithmNTT, N256)      { compare_against_horner(256); }
TEST(CrossAlgorithmNTT, N512)      { compare_against_horner(512); }
