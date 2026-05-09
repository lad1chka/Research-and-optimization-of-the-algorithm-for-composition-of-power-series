// take_even / take_odd projections and their adjoints (interleave with zeros).

#ifndef PSCOMP_TRANSPOSE_INTERLEAVE_HPP
#define PSCOMP_TRANSPOSE_INTERLEAVE_HPP

#include <algorithm>
#include <cstddef>
#include <vector>

#include "pscomp/field/coef_traits.hpp"

namespace pscomp::transpose {

template <class Coef>
std::vector<Coef> interleave_even(const std::vector<Coef>& src, std::size_t out_len) {
    std::vector<Coef> out(out_len, coef_zero<Coef>());
    const std::size_t lim = std::min(src.size(), (out_len + 1) / 2);
    for (std::size_t i = 0; i < lim; ++i) out[2 * i] = src[i];
    return out;
}

template <class Coef>
std::vector<Coef> interleave_odd(const std::vector<Coef>& src, std::size_t out_len) {
    std::vector<Coef> out(out_len, coef_zero<Coef>());
    const std::size_t lim = std::min(src.size(), out_len / 2);
    for (std::size_t i = 0; i < lim; ++i) out[2 * i + 1] = src[i];
    return out;
}

template <class Coef>
std::vector<Coef> interleave_parity(const std::vector<Coef>& src,
                                    std::size_t out_len,
                                    std::size_t parity) {
    return parity == 0 ? interleave_even(src, out_len)
                       : interleave_odd(src, out_len);
}

template <class Coef>
void negate_odd_inplace(std::vector<Coef>& v) {
    for (std::size_t i = 1; i < v.size(); i += 2) v[i] = coef_zero<Coef>() - v[i];
}

template <class Coef>
std::vector<Coef> negate_odd(const std::vector<Coef>& v) {
    std::vector<Coef> out = v;
    negate_odd_inplace(out);
    return out;
}

template <class Coef>
std::vector<Coef> take_even(const std::vector<Coef>& v) {
    std::vector<Coef> out((v.size() + 1) / 2);
    for (std::size_t i = 0; i < out.size(); ++i) out[i] = v[2 * i];
    return out;
}

template <class Coef>
std::vector<Coef> take_odd(const std::vector<Coef>& v) {
    std::vector<Coef> out(v.size() / 2);
    for (std::size_t i = 0; i < out.size(); ++i) out[i] = v[2 * i + 1];
    return out;
}

template <class Coef>
std::vector<Coef> take_parity(const std::vector<Coef>& v, std::size_t parity) {
    return parity == 0 ? take_even(v) : take_odd(v);
}

}  // namespace pscomp::transpose

#endif
