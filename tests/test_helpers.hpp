// Shared utilities for the pscomp test suite: ModInt construction, random
// vector generation against a fixed seed, and the listing of every (algorithm,
// optimization level) entry point that the cross-algorithm checks should
// validate.

#ifndef PSCOMP_TESTS_TEST_HELPERS_HPP
#define PSCOMP_TESTS_TEST_HELPERS_HPP

#include <cstdint>
#include <functional>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "pscomp/pscomp.hpp"

namespace pscomp_tests {

constexpr std::uint32_t kSeed = 0xC0DEBABEu;

inline std::vector<pscomp::ModInt998> random_ntt(std::size_t n, std::mt19937& rng,
                                                 bool g0_zero = false) {
    std::uniform_int_distribution<std::uint32_t> dist(0u, pscomp::kPrime998 - 1u);
    std::vector<pscomp::ModInt998> v(n);
    for (std::size_t i = 0; i < n; ++i) v[i] = pscomp::ModInt998{dist(rng)};
    if (g0_zero && !v.empty()) v[0] = pscomp::ModInt998{};
    return v;
}

template <class Real>
std::vector<Real> random_real(std::size_t n, std::mt19937& rng, bool g0_zero = false) {
    std::uniform_real_distribution<Real> dist(static_cast<Real>(-1), static_cast<Real>(1));
    std::vector<Real> v(n);
    for (std::size_t i = 0; i < n; ++i) v[i] = dist(rng);
    if (g0_zero && !v.empty()) v[0] = Real(0);
    return v;
}

// Catalog of every NTT compose entry point. Returned by the test fixtures so
// each suite iterates over the same set without duplicating the list.
template <class Coef>
struct ComposeEntry {
    std::string                                                        name;
    std::function<std::vector<Coef>(pscomp::span<const Coef>,
                                    pscomp::span<const Coef>,
                                    std::size_t)>                      fn;
};

template <class Coef>
inline std::vector<ComposeEntry<Coef>> all_entries(bool include_naive_def = true) {
    using namespace pscomp::algo;
    std::vector<ComposeEntry<Coef>> out;
    if (include_naive_def) {
        out.push_back({"naive_def",       [](auto f, auto g, std::size_t n) {
            return compose_naive_def<Coef>(f, g, n); }});
    }
    out.push_back({"naive_horner",       [](auto f, auto g, std::size_t n) {
        return compose_naive_horner<Coef>(f, g, n); }});
    out.push_back({"brent_kung_basic",    [](auto f, auto g, std::size_t n) {
        return compose_brent_kung_basic<Coef>(f, g, n); }});
    out.push_back({"brent_kung_opt",      [](auto f, auto g, std::size_t n) {
        return compose_brent_kung_opt<Coef>(f, g, n); }});
    out.push_back({"brent_kung_streaming",[](auto f, auto g, std::size_t n) {
        return compose_brent_kung_streaming<Coef>(f, g, n); }});
    out.push_back({"brent_kung_tuned_m",  [](auto f, auto g, std::size_t n) {
        return compose_brent_kung_tuned_m<Coef>(f, g, n); }});
    out.push_back({"kl_basic",            [](auto f, auto g, std::size_t n) {
        return compose_kl_basic<Coef>(f, g, n); }});
    out.push_back({"kl_inplace",          [](auto f, auto g, std::size_t n) {
        return compose_kl_inplace_bostan_mori<Coef>(f, g, n); }});
    out.push_back({"kl_truncated_mul",    [](auto f, auto g, std::size_t n) {
        return compose_kl_truncated_mul<Coef>(f, g, n); }});
    out.push_back({"kl_packed_pq",        [](auto f, auto g, std::size_t n) {
        return compose_kl_packed_pq<Coef>(f, g, n); }});
    out.push_back({"kl_threshold",        [](auto f, auto g, std::size_t n) {
        return compose_kl_recursion_threshold<Coef>(f, g, n, 32); }});
    return out;
}

}  // namespace pscomp_tests

#endif  // PSCOMP_TESTS_TEST_HELPERS_HPP
