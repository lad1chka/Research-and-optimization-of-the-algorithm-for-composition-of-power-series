# pscomp

Research and optimization of the power-series composition algorithm.
Coursework of Vladislav Moskvitin, Applied Data Analysis and AI, 4th year.

Reference:
> Kinoshita, Li. *Power Series Composition in Near-Linear Time*, arXiv:2404.05177, 2024.

Given formal power series `f` and `g` with `g(0) = 0`, the project computes
the first `n` coefficients of `h(x) = f(g(x))`. The repository contains
header-only C++17 implementations of the naive (`O(n^2)`), Brent-Kung
(`O(sqrt(n) M(n))`) and Kinoshita-Li (`O(n log^2 n)` upper bound) families,
each with paired `_basic` and optimized variants, a GoogleTest suite, and a
Google Benchmark harness.

For derivations and the full optimization list see
[`ALGORITHMS.md`](ALGORITHMS.md).

## Repository layout

```
include/pscomp/
  field/           ModInt998 (Montgomery), ModInt998Plain, coef_traits
  transform/       fft, fft_basic, ntt, ntt_basic
  poly/            poly_mul, poly_utils
  compose/         naive, brent_kung, kinoshita_li, kinoshita_li_tellegen, api
  transpose/       middle_product, interleave, dual_extraction
  workspace/       Arena, CountingResource (std::pmr)
  span.hpp         C++17 substitute for std::span
  pscomp.hpp       umbrella header
src/dev_check.cpp  TU that pulls the umbrella header through warning flags
tests/             GoogleTest correctness and precision suites
benchmarks/        Google Benchmark time and memory binaries
```

## Building

CMake 3.16+ and a C++17 compiler. GoogleTest and Google Benchmark are fetched
via `FetchContent`.

```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Options:

| Option                    | Default | Effect                              |
|---------------------------|---------|-------------------------------------|
| `PSCOMP_BUILD_TESTS`      | `ON`    | Build GoogleTest suites.            |
| `PSCOMP_BUILD_BENCHMARKS` | `ON`    | Build Google Benchmark binaries.    |
| `PSCOMP_NATIVE_ARCH`      | `ON`    | Pass `-march=native` to the compiler. |

## Running the tests

```
ctest --test-dir build --output-on-failure
```

Suites:
* `test_known_series` — closed-form regressions over `ModInt998`.
* `test_cross_algorithms_ntt` — every variant matches `compose_naive_horner`
  on random NTT inputs (n = 16..512).
* `test_random_ntt_large` — optimized variants cross-checked against
  `compose_brent_kung_opt` at n = 1024 and 2048.
* `test_fft_cross_algorithm` — pairwise FFT precision; reports per-pair
  abs/rel error distribution and asserts a per-`n` threshold.
* `test_middle_product`, `test_dual_extraction`, `test_compose_kl_tellegen`
  — Tellegen pipeline.

## Running the benchmarks

```
./build/benchmarks/bench_compose_time   --benchmark_format=csv --benchmark_out=time.csv
./build/benchmarks/bench_compose_memory --benchmark_format=csv --benchmark_out=memory.csv
```

Time benchmarks cover `n in {256, 1024, 4096, 16384, 65536}`. The memory
benchmark reports library allocations (via `pscomp::CountingResource` plugged
into the `std::pmr` default) and process peak RSS (`getrusage`).

## API quick reference

```cpp
#include "pscomp/pscomp.hpp"
using namespace pscomp;

std::vector<ModInt998> f = ...;
std::vector<ModInt998> g = ...;            // g[0] must be 0

ComposeOptions opts;
opts.algo  = Algorithm::KinoshitaLi;
opts.level = OptLevel::Optimized;
auto h = compose<ModInt998>(f, g, 4096, opts);

opts.kl_variant = KLVariant::Tellegen;
auto h_tellegen = compose<ModInt998>(f, g, 4096, opts);
```

The dispatcher routes to the relevant template instantiation; for fine
control call `algo::compose_*` entry points directly. `compose_kl_tellegen`
agrees bit-for-bit with `compose_kl_basic` over `ModInt998`
(`test_compose_kl_tellegen`).

## Composition complexities

| Algorithm                                   | Time                                               | Memory                                  |
|---------------------------------------------|----------------------------------------------------|-----------------------------------------|
| `compose_naive_def`                         | `O(n^2)`                                           | `O(n)`                                  |
| `compose_naive_horner`                      | `O(n^2)`                                           | `O(n)`                                  |
| `compose_brent_kung_*`                      | `O(sqrt(n) * M(n))`                                | `O(n^{1.5})` (`_streaming`: `O(n)`)     |
| `compose_kl_basic`                          | `O(n^2 polylog n)` (schoolbook bi-mul)             | `O(n^2)` BiPoly buffers                 |
| `compose_kl_truncated_mul`                  | `O(n^{1.5} polylog n)`                             | `O(n^{1.5})`                            |
| `compose_kl_recursion_threshold`            | KL above threshold, naive Horner below             | as above                                |
| `compose_kl_tellegen` (`KLVariant::Tellegen`) | `O(n m polylog n)`, same class as `_basic` via middle product | `O(n m)`                       |
| Kinoshita-Li theoretical optimum            | `O(n log^2 n)` (requires y-axis Bostan-Mori, §8.6) | `O(n log n)`                            |

## CI

`.github/workflows/ci.yml` runs the GoogleTest suite on Ubuntu and macOS,
Debug and Release.
