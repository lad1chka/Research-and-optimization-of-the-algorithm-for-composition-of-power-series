// Flat (row-major) representation of a bivariate polynomial.

#ifndef PSCOMP_TRANSPOSE_FLAT_BIPOLY_HPP
#define PSCOMP_TRANSPOSE_FLAT_BIPOLY_HPP

#include <cstddef>
#include <vector>

#include "pscomp/field/coef_traits.hpp"

namespace pscomp::transpose {

template <class Coef>
struct FlatBi {
    std::vector<Coef> data;
    std::size_t n_x = 0;
    std::size_t n_y = 0;

    FlatBi() = default;
    FlatBi(std::size_t nx, std::size_t ny)
        : data(nx * ny, coef_zero<Coef>()), n_x(nx), n_y(ny) {}

    Coef&       at(std::size_t i, std::size_t j)       { return data[i * n_y + j]; }
    const Coef& at(std::size_t i, std::size_t j) const { return data[i * n_y + j]; }
    Coef*       row(std::size_t i)       { return data.data() + i * n_y; }
    const Coef* row(std::size_t i) const { return data.data() + i * n_y; }
};

}  // namespace pscomp::transpose

#endif
