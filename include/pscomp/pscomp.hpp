// Umbrella header. Pulls in everything required to compose two formal power
// series and benchmark the underlying primitives.

#ifndef PSCOMP_PSCOMP_HPP
#define PSCOMP_PSCOMP_HPP

#include "pscomp/compose/api.hpp"
#include "pscomp/compose/brent_kung.hpp"
#include "pscomp/compose/kinoshita_li.hpp"
#include "pscomp/compose/naive.hpp"
#include "pscomp/field/coef_traits.hpp"
#include "pscomp/field/mod_int.hpp"
#include "pscomp/poly/poly_mul.hpp"
#include "pscomp/poly/poly_utils.hpp"
#include "pscomp/span.hpp"
#include "pscomp/transform/fft.hpp"
#include "pscomp/transform/fft_basic.hpp"
#include "pscomp/transform/ntt.hpp"
#include "pscomp/transform/ntt_basic.hpp"
#include "pscomp/workspace/arena.hpp"
#include "pscomp/workspace/counting_resource.hpp"

#endif  // PSCOMP_PSCOMP_HPP
