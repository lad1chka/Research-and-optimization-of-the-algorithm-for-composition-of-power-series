// Composition dispatcher.

#ifndef PSCOMP_COMPOSE_API_HPP
#define PSCOMP_COMPOSE_API_HPP

#include <cstddef>
#include <stdexcept>
#include <vector>

#include "pscomp/compose/brent_kung.hpp"
#include "pscomp/compose/kinoshita_li.hpp"
#include "pscomp/compose/kinoshita_li_tellegen.hpp"
#include "pscomp/compose/naive.hpp"
#include "pscomp/span.hpp"

namespace pscomp {

enum class Algorithm {
    NaiveDef,
    NaiveHorner,
    BrentKung,
    KinoshitaLi
};

enum class OptLevel {
    Basic,
    Optimized
};

enum class KLVariant {
    Forward,
    Tellegen
};

struct ComposeOptions {
    Algorithm algo              = Algorithm::KinoshitaLi;
    OptLevel  level             = OptLevel::Optimized;
    KLVariant kl_variant        = KLVariant::Forward;
    std::size_t naive_threshold = 64;
};

template <class Coef>
std::vector<Coef> compose(span<const Coef> f,
                          span<const Coef> g,
                          std::size_t n,
                          ComposeOptions opts = {}) {
    using namespace pscomp::algo;
    switch (opts.algo) {
        case Algorithm::NaiveDef:
            return compose_naive_def<Coef>(f, g, n);
        case Algorithm::NaiveHorner:
            return compose_naive_horner<Coef>(f, g, n);
        case Algorithm::BrentKung:
            switch (opts.level) {
                case OptLevel::Basic:     return compose_brent_kung_basic<Coef>(f, g, n);
                case OptLevel::Optimized: return compose_brent_kung_opt<Coef>(f, g, n);
            }
            break;
        case Algorithm::KinoshitaLi:
            if (opts.kl_variant == KLVariant::Tellegen) {
                return compose_kl_tellegen<Coef>(f, g, n);
            }
            switch (opts.level) {
                case OptLevel::Basic:     return compose_kl_basic<Coef>(f, g, n);
                case OptLevel::Optimized: return compose_kl_recursion_threshold<Coef>(
                                                  f, g, n, opts.naive_threshold);
            }
            break;
    }
    throw std::invalid_argument("pscomp::compose: invalid options");
}

}  // namespace pscomp

#endif
