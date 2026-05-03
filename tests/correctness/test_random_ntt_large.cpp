// Stress test that the *_opt NTT-backed variants stay exact at sizes where
// quadratic baselines are too slow to act as oracles. Brent-Kung optimized
// becomes the reference, justified because the prior small-N tests already
// pin every variant against naive Horner.

#include <gtest/gtest.h>

#include "test_helpers.hpp"

using pscomp::ModInt998;
using namespace pscomp_tests;

namespace {

void cross_check_large(std::size_t n) {
    std::mt19937 rng(kSeed ^ static_cast<std::uint32_t>(n + 17));
    auto f = random_ntt(n, rng);
    auto g = random_ntt(n, rng, /*g0_zero=*/true);

    auto reference = pscomp::algo::compose_brent_kung_opt<ModInt998>(f, g, n);

    auto kl = pscomp::algo::compose_kl_truncated_mul<ModInt998>(f, g, n);
    ASSERT_EQ(kl.size(), reference.size());
    for (std::size_t i = 0; i < n; ++i) {
        EXPECT_EQ(kl[i], reference[i]) << "kl_truncated_mul n=" << n << " i=" << i;
    }

    auto bk_streaming = pscomp::algo::compose_brent_kung_streaming<ModInt998>(f, g, n);
    for (std::size_t i = 0; i < n; ++i) {
        EXPECT_EQ(bk_streaming[i], reference[i])
            << "bk_streaming n=" << n << " i=" << i;
    }

    auto bk_tuned = pscomp::algo::compose_brent_kung_tuned_m<ModInt998>(f, g, n);
    for (std::size_t i = 0; i < n; ++i) {
        EXPECT_EQ(bk_tuned[i], reference[i])
            << "bk_tuned n=" << n << " i=" << i;
    }
}

}  // namespace

TEST(LargeNTT, N1024) { cross_check_large(1024); }
TEST(LargeNTT, N2048) { cross_check_large(2048); }
