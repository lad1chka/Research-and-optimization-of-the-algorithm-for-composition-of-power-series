// Closed-form regression tests: evaluate composition against power series
// whose coefficients are known analytically (or by hand calculation).

#include <gtest/gtest.h>

#include "test_helpers.hpp"

using pscomp::ModInt998;
using pscomp::span;
using namespace pscomp_tests;

namespace {

std::vector<ModInt998> as_mod(std::initializer_list<int> xs) {
    std::vector<ModInt998> out;
    out.reserve(xs.size());
    for (int x : xs) {
        out.push_back(x >= 0 ? ModInt998{static_cast<std::uint64_t>(x)}
                              : -ModInt998{static_cast<std::uint64_t>(-x)});
    }
    return out;
}

void expect_equal(const std::vector<ModInt998>& got,
                  const std::vector<ModInt998>& expected,
                  const std::string& label) {
    ASSERT_EQ(got.size(), expected.size()) << label;
    for (std::size_t i = 0; i < got.size(); ++i) {
        EXPECT_EQ(got[i], expected[i]) << label << " coef[" << i << "]";
    }
}

}  // namespace

// f(g(x)) where f = 1 + 2x + 3x^2 and g = x + x^2 expands to 1 + 2x + 5x^2 + 6x^3
// modulo x^4. Hand-verified.
TEST(KnownSeriesNTT, Quadratic) {
    auto f        = as_mod({1, 2, 3});
    auto g        = as_mod({0, 1, 1});
    auto expected = as_mod({1, 2, 5, 6});

    for (auto& entry : all_entries<ModInt998>(/*include_naive_def=*/true)) {
        auto out = entry.fn(f, g, expected.size());
        expect_equal(out, expected, entry.name);
    }
}

// Composition with the identity polynomial yields f truncated to n terms.
TEST(KnownSeriesNTT, IdentityComposition) {
    auto f = as_mod({7, 3, 11, 1, 9, 4});
    auto g = as_mod({0, 1});
    for (auto& entry : all_entries<ModInt998>()) {
        auto out = entry.fn(f, g, f.size());
        expect_equal(out, f, entry.name);
    }
}

// Constant f returns the constant truncated to n.
TEST(KnownSeriesNTT, ConstantF) {
    auto f = as_mod({42});
    auto g = as_mod({0, 1, 5, 7, 9});
    std::vector<ModInt998> expected(g.size(), ModInt998{});
    expected[0] = ModInt998{42};
    for (auto& entry : all_entries<ModInt998>()) {
        auto out = entry.fn(f, g, g.size());
        expect_equal(out, expected, entry.name);
    }
}

// f(g) where g is purely linear bx => f(bx).
TEST(KnownSeriesNTT, LinearScale) {
    const ModInt998 b{3u};
    auto f = as_mod({1, 1, 1, 1, 1});
    std::vector<ModInt998> g{ModInt998{}, b};
    std::vector<ModInt998> expected(f.size(), ModInt998{});
    ModInt998 power{1u};
    for (std::size_t i = 0; i < f.size(); ++i) {
        expected[i] = f[i] * power;
        power = power * b;
    }
    for (auto& entry : all_entries<ModInt998>()) {
        auto out = entry.fn(f, g, f.size());
        expect_equal(out, expected, entry.name);
    }
}

// (1 + x)^k composed with x maps trivially: verify against pre-expanded
// binomial coefficients for k = 4.
TEST(KnownSeriesNTT, Binomial) {
    auto f        = as_mod({1, 4, 6, 4, 1});  // (1 + y)^4
    auto g        = as_mod({0, 1, 0, 0, 0});  // y -> x
    auto expected = as_mod({1, 4, 6, 4, 1});
    for (auto& entry : all_entries<ModInt998>()) {
        auto out = entry.fn(f, g, expected.size());
        expect_equal(out, expected, entry.name);
    }
}
