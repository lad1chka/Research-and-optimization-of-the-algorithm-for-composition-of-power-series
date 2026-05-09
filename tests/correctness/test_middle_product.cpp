// Tests the middle product: adjoint identity, backend agreement, edge cases.

#include <gtest/gtest.h>

#include <cstdint>
#include <random>
#include <vector>

#include "pscomp/pscomp.hpp"
#include "pscomp/transpose/middle_product.hpp"
#include "test_helpers.hpp"

using pscomp::ModInt998;
using pscomp::ModInt998Plain;
using pscomp::span;
using pscomp::transpose::middle_product;
using pscomp::transpose::mp_fft;
using pscomp::transpose::mp_ntt;
using pscomp::transpose::mp_ntt_basic;
using pscomp::transpose::mp_schoolbook;

namespace {

template <class Coef>
std::vector<Coef> schoolbook_mul(span<const Coef> a, span<const Coef> b) {
    if (a.empty() || b.empty()) return {};
    std::vector<Coef> out(a.size() + b.size() - 1, pscomp::coef_zero<Coef>());
    for (std::size_t i = 0; i < a.size(); ++i) {
        for (std::size_t j = 0; j < b.size(); ++j) {
            out[i + j] = out[i + j] + a[i] * b[j];
        }
    }
    return out;
}

template <class Coef>
std::vector<Coef> random_mod(std::size_t n, std::mt19937& rng) {
    std::uniform_int_distribution<std::uint32_t> dist(0u, pscomp::kPrime998 - 1u);
    std::vector<Coef> v(n);
    for (auto& x : v) x = Coef{dist(rng)};
    return v;
}

}  // namespace

TEST(MiddleProduct, AdjointIdentityModInt) {
    std::mt19937 rng(pscomp_tests::kSeed ^ 0x11u);
    for (std::size_t n : {1u, 2u, 5u, 17u, 64u}) {
        for (std::size_t m : {1u, 2u, 8u, 33u}) {
            auto a = random_mod<ModInt998>(n, rng);
            auto b = random_mod<ModInt998>(m, rng);
            const std::size_t x_len = n + m - 1;
            std::vector<ModInt998> x(x_len);
            std::uniform_int_distribution<std::uint32_t> dist(0u, pscomp::kPrime998 - 1u);
            for (auto& xi : x) xi = ModInt998{dist(rng)};

            auto mp = mp_schoolbook<ModInt998>(b, x, n);
            auto ab = schoolbook_mul<ModInt998>(a, b);
            ASSERT_EQ(ab.size(), x.size());

            ModInt998 lhs = pscomp::coef_zero<ModInt998>();
            for (std::size_t k = 0; k < n; ++k) lhs = lhs + mp[k] * a[k];

            ModInt998 rhs = pscomp::coef_zero<ModInt998>();
            for (std::size_t i = 0; i < x.size(); ++i) rhs = rhs + x[i] * ab[i];

            EXPECT_EQ(lhs.to_uint(), rhs.to_uint())
                << "n=" << n << " m=" << m;
        }
    }
}

TEST(MiddleProduct, BackendAgreementNTT) {
    std::mt19937 rng(pscomp_tests::kSeed ^ 0x22u);
    for (std::size_t m : {2u, 8u, 33u, 128u}) {
        for (std::size_t n_out : {1u, 16u, 100u, 257u}) {
            const std::size_t x_len = n_out + m - 1;
            auto b = random_mod<ModInt998>(m, rng);
            auto x = random_mod<ModInt998>(x_len, rng);

            auto ref = mp_schoolbook<ModInt998>(b, x, n_out);
            auto opt = mp_ntt(b, x, n_out);
            ASSERT_EQ(opt.size(), ref.size());
            for (std::size_t i = 0; i < n_out; ++i) {
                EXPECT_EQ(opt[i].to_uint(), ref[i].to_uint())
                    << "m=" << m << " n_out=" << n_out << " i=" << i;
            }
        }
    }
}

TEST(MiddleProduct, BackendAgreementNTTBasic) {
    std::mt19937 rng(pscomp_tests::kSeed ^ 0x33u);
    for (std::size_t m : {2u, 8u, 33u, 128u}) {
        for (std::size_t n_out : {1u, 16u, 100u, 257u}) {
            const std::size_t x_len = n_out + m - 1;
            auto b = random_mod<ModInt998Plain>(m, rng);
            auto x = random_mod<ModInt998Plain>(x_len, rng);

            auto ref = mp_schoolbook<ModInt998Plain>(b, x, n_out);
            auto opt = mp_ntt_basic(b, x, n_out);
            ASSERT_EQ(opt.size(), ref.size());
            for (std::size_t i = 0; i < n_out; ++i) {
                EXPECT_EQ(opt[i].to_uint(), ref[i].to_uint())
                    << "m=" << m << " n_out=" << n_out << " i=" << i;
            }
        }
    }
}

TEST(MiddleProduct, BackendAgreementFFT) {
    std::mt19937 rng(pscomp_tests::kSeed ^ 0x44u);
    for (std::size_t m : {2u, 8u, 33u, 128u}) {
        for (std::size_t n_out : {1u, 16u, 100u, 257u}) {
            const std::size_t x_len = n_out + m - 1;
            auto b = pscomp_tests::random_real<double>(m, rng);
            auto x = pscomp_tests::random_real<double>(x_len, rng);

            auto ref = mp_schoolbook<double>(b, x, n_out);
            auto opt = mp_fft<double>(b, x, n_out);
            ASSERT_EQ(opt.size(), ref.size());
            const double tol = 1e-10 * static_cast<double>(m);
            for (std::size_t i = 0; i < n_out; ++i) {
                EXPECT_NEAR(opt[i], ref[i], tol)
                    << "m=" << m << " n_out=" << n_out << " i=" << i;
            }
        }
    }
}

TEST(MiddleProduct, DispatcherSelectsBackend) {
    std::mt19937 rng(pscomp_tests::kSeed ^ 0x55u);
    auto bm = random_mod<ModInt998>(7, rng);
    auto xm = random_mod<ModInt998>(20, rng);
    auto via_dispatch = middle_product<ModInt998>(bm, xm, 14);
    auto via_explicit = mp_ntt(bm, xm, 14);
    for (std::size_t i = 0; i < via_dispatch.size(); ++i) {
        EXPECT_EQ(via_dispatch[i].to_uint(), via_explicit[i].to_uint());
    }

    auto br = pscomp_tests::random_real<double>(7, rng);
    auto xr = pscomp_tests::random_real<double>(20, rng);
    auto rdispatch = middle_product<double>(br, xr, 14);
    auto rexplicit = mp_fft<double>(br, xr, 14);
    for (std::size_t i = 0; i < rdispatch.size(); ++i) {
        EXPECT_NEAR(rdispatch[i], rexplicit[i], 1e-12);
    }
}

TEST(MiddleProduct, EdgeCases) {
    std::vector<ModInt998> empty;
    auto e1 = mp_schoolbook<ModInt998>(span<const ModInt998>(empty),
                                        span<const ModInt998>(empty), 4);
    EXPECT_EQ(e1.size(), 4u);
    for (auto v : e1) EXPECT_EQ(v.to_uint(), 0u);

    std::vector<ModInt998> b1 = {ModInt998{7u}};
    std::vector<ModInt998> x = {ModInt998{1u}, ModInt998{2u}, ModInt998{3u}};
    auto out = mp_schoolbook<ModInt998>(b1, x, 3);
    EXPECT_EQ(out[0].to_uint(), 7u);
    EXPECT_EQ(out[1].to_uint(), 14u);
    EXPECT_EQ(out[2].to_uint(), 21u);

    auto z = mp_ntt(b1, x, 0);
    EXPECT_TRUE(z.empty());
}
