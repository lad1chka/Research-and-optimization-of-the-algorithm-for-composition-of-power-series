// Per-pair error distribution (min/max/mean/var/quantiles, abs and rel).

#ifndef PSCOMP_TESTS_PRECISION_ERROR_STATS_HPP
#define PSCOMP_TESTS_PRECISION_ERROR_STATS_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <ostream>
#include <vector>

namespace pscomp_tests {

struct ErrorStats {
    double min{0.0};
    double max{0.0};
    double mean{0.0};
    double variance{0.0};
    double q10{0.0}, q25{0.0}, q50{0.0}, q75{0.0}, q90{0.0};
    double rel_max{0.0};
    double rel_mean{0.0};
    double rel_q90{0.0};
    double max_abs_value{0.0};
    std::size_t count{0};
};

inline double quantile(std::vector<double>& sorted, double q) {
    if (sorted.empty()) return 0.0;
    const double pos = q * static_cast<double>(sorted.size() - 1);
    const std::size_t lo = static_cast<std::size_t>(std::floor(pos));
    const std::size_t hi = static_cast<std::size_t>(std::ceil(pos));
    const double frac = pos - static_cast<double>(lo);
    return sorted[lo] + (sorted[hi] - sorted[lo]) * frac;
}

template <class Real>
ErrorStats compute_error_stats(const std::vector<Real>& a, const std::vector<Real>& b,
                                double rel_floor = 1.0) {
    ErrorStats s;
    const std::size_t n = std::min(a.size(), b.size());
    if (n == 0) return s;

    std::vector<double> deltas;
    std::vector<double> rel_deltas;
    deltas.reserve(n);
    rel_deltas.reserve(n);
    double sum = 0.0;
    double rel_sum = 0.0;
    double max_abs = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double ai = static_cast<double>(a[i]);
        const double bi = static_cast<double>(b[i]);
        const double d  = std::abs(ai - bi);
        deltas.push_back(d);
        sum += d;

        const double scale = std::max({std::abs(ai), std::abs(bi), rel_floor});
        const double rd    = d / scale;
        rel_deltas.push_back(rd);
        rel_sum += rd;

        max_abs = std::max({max_abs, std::abs(ai), std::abs(bi)});
    }
    s.count = n;
    s.mean  = sum / static_cast<double>(n);
    s.rel_mean = rel_sum / static_cast<double>(n);
    s.max_abs_value = max_abs;

    double sse = 0.0;
    for (double d : deltas) {
        const double diff = d - s.mean;
        sse += diff * diff;
    }
    s.variance = sse / static_cast<double>(n);

    std::sort(deltas.begin(), deltas.end());
    s.min = deltas.front();
    s.max = deltas.back();
    s.q10 = quantile(deltas, 0.10);
    s.q25 = quantile(deltas, 0.25);
    s.q50 = quantile(deltas, 0.50);
    s.q75 = quantile(deltas, 0.75);
    s.q90 = quantile(deltas, 0.90);

    std::sort(rel_deltas.begin(), rel_deltas.end());
    s.rel_max = rel_deltas.back();
    s.rel_q90 = quantile(rel_deltas, 0.90);
    return s;
}

inline std::ostream& operator<<(std::ostream& os, const ErrorStats& s) {
    return os << "n=" << s.count
              << " |max|=" << s.max_abs_value
              << " abs{min=" << s.min << " max=" << s.max
              << " mean=" << s.mean << " var=" << s.variance
              << " q10=" << s.q10 << " q25=" << s.q25
              << " q50=" << s.q50 << " q75=" << s.q75
              << " q90=" << s.q90 << "}"
              << " rel{max=" << s.rel_max
              << " mean=" << s.rel_mean
              << " q90=" << s.rel_q90 << "}";
}

}  // namespace pscomp_tests

#endif
