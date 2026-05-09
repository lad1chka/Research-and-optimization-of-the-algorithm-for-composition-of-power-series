// Modular integers over the NTT prime 998244353.

#ifndef PSCOMP_FIELD_MOD_INT_HPP
#define PSCOMP_FIELD_MOD_INT_HPP

#include <cstdint>
#include <ostream>

namespace pscomp {

inline constexpr std::uint32_t kPrime998      = 998244353u;
inline constexpr std::uint32_t kPrime998Root  = 3u;
inline constexpr std::uint32_t kPrime998Order = 1u << 23;

namespace detail {

constexpr std::uint32_t mulmod32(std::uint32_t a, std::uint32_t b) noexcept {
    return static_cast<std::uint32_t>(static_cast<std::uint64_t>(a) * b % kPrime998);
}

constexpr std::uint32_t powmod32(std::uint32_t a, std::uint64_t e) noexcept {
    std::uint32_t r = 1u;
    while (e) {
        if (e & 1u) r = mulmod32(r, a);
        a = mulmod32(a, a);
        e >>= 1;
    }
    return r;
}

// -P^{-1} mod 2^32 via Newton iteration.
constexpr std::uint32_t mont_ninv() noexcept {
    std::uint32_t x = 1u;
    for (int i = 0; i < 5; ++i) x *= 2u - kPrime998 * x;
    return ~x + 1u;
}

constexpr std::uint32_t mont_r1() noexcept {
    return static_cast<std::uint32_t>((1ull << 32) % kPrime998);
}

constexpr std::uint32_t mont_r2() noexcept {
    const std::uint64_t r = mont_r1();
    return static_cast<std::uint32_t>((r * r) % kPrime998);
}

inline constexpr std::uint32_t kMontNInv = mont_ninv();
inline constexpr std::uint32_t kMontR1   = mont_r1();
inline constexpr std::uint32_t kMontR2   = mont_r2();

// (a / R) mod P with a < P * R.
inline std::uint32_t mont_reduce(std::uint64_t a) noexcept {
    const std::uint32_t m = static_cast<std::uint32_t>(a) * kMontNInv;
    const std::uint64_t t = (a + static_cast<std::uint64_t>(m) * kPrime998) >> 32;
    return static_cast<std::uint32_t>(t < kPrime998 ? t : t - kPrime998);
}

}  // namespace detail

class ModInt998 {
public:
    constexpr ModInt998() noexcept = default;
    explicit ModInt998(std::uint64_t x) noexcept
        : v_(detail::mont_reduce(static_cast<std::uint64_t>(static_cast<std::uint32_t>(x % kPrime998)) *
                                 detail::kMontR2)) {}

    static ModInt998 from_raw(std::uint32_t mont_value) noexcept {
        ModInt998 r;
        r.v_ = mont_value;
        return r;
    }

    std::uint32_t to_uint() const noexcept { return detail::mont_reduce(v_); }
    std::uint32_t raw()     const noexcept { return v_; }

    ModInt998& operator+=(ModInt998 o) noexcept {
        v_ += o.v_;
        if (v_ >= kPrime998) v_ -= kPrime998;
        return *this;
    }
    ModInt998& operator-=(ModInt998 o) noexcept {
        v_ = (v_ >= o.v_) ? (v_ - o.v_) : (v_ + kPrime998 - o.v_);
        return *this;
    }
    ModInt998& operator*=(ModInt998 o) noexcept {
        v_ = detail::mont_reduce(static_cast<std::uint64_t>(v_) * o.v_);
        return *this;
    }

    friend ModInt998 operator+(ModInt998 a, ModInt998 b) noexcept { return a += b; }
    friend ModInt998 operator-(ModInt998 a, ModInt998 b) noexcept { return a -= b; }
    friend ModInt998 operator*(ModInt998 a, ModInt998 b) noexcept { return a *= b; }
    friend ModInt998 operator-(ModInt998 a) noexcept {
        return a.v_ == 0 ? a : ModInt998::from_raw(kPrime998 - a.v_);
    }
    friend bool operator==(ModInt998 a, ModInt998 b) noexcept { return a.v_ == b.v_; }
    friend bool operator!=(ModInt998 a, ModInt998 b) noexcept { return a.v_ != b.v_; }

    ModInt998 pow(std::uint64_t e) const noexcept {
        ModInt998 r = ModInt998::from_raw(detail::kMontR1);
        ModInt998 b = *this;
        while (e) {
            if (e & 1u) r *= b;
            b *= b;
            e >>= 1;
        }
        return r;
    }
    ModInt998 inv() const noexcept { return pow(kPrime998 - 2); }

    friend std::ostream& operator<<(std::ostream& os, ModInt998 x) {
        return os << x.to_uint();
    }

private:
    std::uint32_t v_ = 0;
};

class ModInt998Plain {
public:
    constexpr ModInt998Plain() noexcept = default;
    constexpr explicit ModInt998Plain(std::uint64_t x) noexcept
        : v_(static_cast<std::uint32_t>(x % kPrime998)) {}

    constexpr std::uint32_t to_uint() const noexcept { return v_; }

    ModInt998Plain& operator+=(ModInt998Plain o) noexcept {
        v_ += o.v_;
        if (v_ >= kPrime998) v_ -= kPrime998;
        return *this;
    }
    ModInt998Plain& operator-=(ModInt998Plain o) noexcept {
        v_ = (v_ >= o.v_) ? (v_ - o.v_) : (v_ + kPrime998 - o.v_);
        return *this;
    }
    ModInt998Plain& operator*=(ModInt998Plain o) noexcept {
        v_ = detail::mulmod32(v_, o.v_);
        return *this;
    }

    friend ModInt998Plain operator+(ModInt998Plain a, ModInt998Plain b) noexcept { return a += b; }
    friend ModInt998Plain operator-(ModInt998Plain a, ModInt998Plain b) noexcept { return a -= b; }
    friend ModInt998Plain operator*(ModInt998Plain a, ModInt998Plain b) noexcept { return a *= b; }
    friend ModInt998Plain operator-(ModInt998Plain a) noexcept {
        return a.v_ == 0 ? a : ModInt998Plain{kPrime998 - a.v_};
    }
    friend bool operator==(ModInt998Plain a, ModInt998Plain b) noexcept { return a.v_ == b.v_; }
    friend bool operator!=(ModInt998Plain a, ModInt998Plain b) noexcept { return a.v_ != b.v_; }

    ModInt998Plain pow(std::uint64_t e) const noexcept {
        ModInt998Plain r{1u}, b = *this;
        while (e) {
            if (e & 1u) r *= b;
            b *= b;
            e >>= 1;
        }
        return r;
    }
    ModInt998Plain inv() const noexcept { return pow(kPrime998 - 2); }

private:
    std::uint32_t v_ = 0;
};

}  // namespace pscomp

#endif
