// Hand-computed compositions over ModInt998.

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

TEST(KnownSeriesNTT, Quadratic) {
    auto f        = as_mod({1, 2, 3});
    auto g        = as_mod({0, 1, 1});
    auto expected = as_mod({1, 2, 5, 6});

    for (auto& entry : all_entries<ModInt998>(true)) {
        auto out = entry.fn(f, g, expected.size());
        expect_equal(out, expected, entry.name);
    }
}

TEST(KnownSeriesNTT, IdentityComposition) {
    auto f = as_mod({7, 3, 11, 1, 9, 4});
    auto g = as_mod({0, 1});
    for (auto& entry : all_entries<ModInt998>()) {
        auto out = entry.fn(f, g, f.size());
        expect_equal(out, f, entry.name);
    }
}

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

TEST(KnownSeriesNTT, Binomial) {
    auto f        = as_mod({1, 4, 6, 4, 1});
    auto g        = as_mod({0, 1, 0, 0, 0});
    auto expected = as_mod({1, 4, 6, 4, 1});
    for (auto& entry : all_entries<ModInt998>()) {
        auto out = entry.fn(f, g, expected.size());
        expect_equal(out, expected, entry.name);
    }
}
