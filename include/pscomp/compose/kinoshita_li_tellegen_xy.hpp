#ifndef PSCOMP_COMPOSE_KINOSHITA_LI_TELLEGEN_XY_HPP
#define PSCOMP_COMPOSE_KINOSHITA_LI_TELLEGEN_XY_HPP

#include <cstddef>
#include <vector>

#include "pscomp/compose/kinoshita_li_tellegen.hpp"
#include "pscomp/span.hpp"

namespace pscomp::algo {

template <class Coef>
std::vector<Coef> compose_kl_tellegen_xy(span<const Coef> f,
                                         span<const Coef> g,
                                         std::size_t n) {
    return compose_kl_tellegen<Coef>(f, g, n);
}

}  // namespace pscomp::algo

#endif
