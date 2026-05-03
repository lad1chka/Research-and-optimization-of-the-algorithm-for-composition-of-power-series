// Coefficient introspection helpers shared by the polynomial and composition
// layers. Mostly distinguishes "is this a modular field?" so that the FFT vs
// NTT backend dispatch can happen at compile time.

#ifndef PSCOMP_FIELD_COEF_TRAITS_HPP
#define PSCOMP_FIELD_COEF_TRAITS_HPP

#include <type_traits>

#include "pscomp/field/mod_int.hpp"

namespace pscomp {

template <class Coef>
struct is_mod_int : std::false_type {};

template <> struct is_mod_int<ModInt998>      : std::true_type {};
template <> struct is_mod_int<ModInt998Plain> : std::true_type {};

template <class Coef>
inline constexpr bool is_mod_int_v = is_mod_int<Coef>::value;

template <class Coef>
struct is_real : std::false_type {};

template <> struct is_real<float>       : std::true_type {};
template <> struct is_real<double>      : std::true_type {};
template <> struct is_real<long double> : std::true_type {};

template <class Coef>
inline constexpr bool is_real_v = is_real<Coef>::value;

template <class Coef>
constexpr Coef coef_zero() noexcept {
    if constexpr (is_mod_int_v<Coef>) return Coef{};
    else                              return Coef(0);
}

template <class Coef>
constexpr Coef coef_one() noexcept {
    if constexpr (is_mod_int_v<Coef>) return Coef{1u};
    else                              return Coef(1);
}

}  // namespace pscomp

#endif  // PSCOMP_FIELD_COEF_TRAITS_HPP
