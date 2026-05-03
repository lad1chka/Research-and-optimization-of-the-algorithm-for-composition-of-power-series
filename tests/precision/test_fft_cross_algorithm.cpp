// FFT precision test: cross-compare every composition variant on real-valued
// coefficients in [-1, 1]. Since no algorithm is treated as ground truth, each
// pair contributes a coefficient-wise delta sample; the resulting distribution
// is summarised (abs/rel min/max/mean/var, quantiles 0.1..0.9) and asserted
// against an empirical RELATIVE worst-case threshold. Absolute deltas alone
// are meaningless here: for random input polynomials the composition
// coefficients can span many orders of magnitude, so a consistent bound is
// only possible after normalising by max(|a_i|, |b_i|).

#include <gtest/gtest.h>

#include <iostream>
#include <random>
#include <vector>

#include "error_stats.hpp"
#include "test_helpers.hpp"

using namespace pscomp_tests;

namespace {

template <class Real>
void run_pairwise_check(std::size_t n, double rel_threshold) {
    std::mt19937 rng(kSeed ^ static_cast<std::uint32_t>(n) ^ 0xFFu);
    auto f = random_real<Real>(n, rng);
    auto g = random_real<Real>(n, rng, /*g0_zero=*/true);
    // Heavy rescaling of g keeps the composition coefficients in a
    // representable range for double; otherwise random polynomials quickly
    // produce astronomical values where rounding swamps meaningful precision.
    const Real scale = Real(0.1);
    for (auto& c : g) c *= scale;

    auto entries = all_entries<Real>(/*include_naive_def=*/n <= 64);

    std::vector<std::vector<Real>> outputs;
    outputs.reserve(entries.size());
    for (auto& e : entries) outputs.push_back(e.fn(f, g, n));

    bool any_failure = false;
    for (std::size_t i = 0; i < entries.size(); ++i) {
        for (std::size_t j = i + 1; j < entries.size(); ++j) {
            auto stats = compute_error_stats(outputs[i], outputs[j]);
            std::cout << "[FFT n=" << n << "] " << entries[i].name
                      << " vs " << entries[j].name << ": " << stats << "\n";
            if (stats.rel_max > rel_threshold) {
                any_failure = true;
                ADD_FAILURE() << entries[i].name << " vs " << entries[j].name
                              << " rel_max=" << stats.rel_max
                              << " exceeds threshold " << rel_threshold;
            }
        }
    }
    EXPECT_FALSE(any_failure);
}

}  // namespace

// Thresholds are on the RELATIVE error max_i |a_i - b_i| / max(|a_i|, |b_i|, 1)
// between any pair of composition variants. On Apple Silicon `long double`
// aliases to `double`, so the LongDouble cases cannot go below the double
// bounds in practice; the separate test suite still exercises the long-double
// code paths for portability.
TEST(FFTPrecision, DoubleN64)   { run_pairwise_check<double>(64,   1e-10); }
TEST(FFTPrecision, DoubleN256)  { run_pairwise_check<double>(256,  1e-8);  }
TEST(FFTPrecision, DoubleN1024) { run_pairwise_check<double>(1024, 1e-6);  }

TEST(FFTPrecisionLD, LongDoubleN64)   { run_pairwise_check<long double>(64,   1e-10); }
TEST(FFTPrecisionLD, LongDoubleN256)  { run_pairwise_check<long double>(256,  1e-8);  }
TEST(FFTPrecisionLD, LongDoubleN1024) { run_pairwise_check<long double>(1024, 1e-6);  }
