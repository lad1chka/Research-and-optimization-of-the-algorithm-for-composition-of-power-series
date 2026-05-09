// Pairwise FFT precision check: compare every compose variant against every
// other variant on random inputs, assert relative max error per n.

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
    auto g = random_real<Real>(n, rng, true);
    const Real scale = Real(0.1);
    for (auto& c : g) c *= scale;

    auto entries = all_entries<Real>(n <= 64);

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

TEST(FFTPrecision, DoubleN64)   { run_pairwise_check<double>(64,   1e-10); }
TEST(FFTPrecision, DoubleN256)  { run_pairwise_check<double>(256,  1e-8);  }
TEST(FFTPrecision, DoubleN1024) { run_pairwise_check<double>(1024, 1e-5);  }

TEST(FFTPrecisionLD, LongDoubleN64)   { run_pairwise_check<long double>(64,   1e-10); }
TEST(FFTPrecisionLD, LongDoubleN256)  { run_pairwise_check<long double>(256,  1e-8);  }
TEST(FFTPrecisionLD, LongDoubleN1024) { run_pairwise_check<long double>(1024, 1e-5);  }
