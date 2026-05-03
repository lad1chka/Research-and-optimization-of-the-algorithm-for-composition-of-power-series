# pscomp

**Research and optimization of the power-series composition algorithm.**
Coursework project of Vladislav Moskvitin, Applied Data Analysis and AI, 4th year.

The work follows
> Kinoshita, Li. *Power Series Composition in Near-Linear Time*, arXiv:2404.05177, 2024.

Power-series composition is the problem of computing the first `n`
coefficients of `h(x) = f(g(x))`, where `f` and `g` are formal power series
and `g(0) = 0`. Classical algorithms run in `O(n^2)`; modern algorithms reach
`O(n^{1.5})` (Brent-Kung, 1978) and `O(n log^2 n)` (Kinoshita-Li, 2024). This
repository ships header-only C++17 implementations of all three families,
with a paired `_basic`/`_opt` version of every component, a GoogleTest-based
correctness suite (including FFT precision quantiles), and a Google
Benchmark harness for time and memory.

For the full algorithmic description (notation, derivations, optimization
list, FFT error model, Tellegen-transposition discussion for KL) see
[`ALGORITHMS.md`](ALGORITHMS.md).

## Repository layout

```
include/pscomp/
  field/           ModInt998 (Montgomery), ModInt998Plain, coef_traits
  transform/       fft, fft_basic, ntt, ntt_basic
  poly/            poly_mul, poly_utils
  compose/         naive, brent_kung, kinoshita_li, api
  workspace/       Arena, CountingResource (std::pmr)
  span.hpp         C++17 substitute for std::span
  pscomp.hpp       umbrella header
src/dev_check.cpp  TU that pulls the umbrella header through the warning flags
tests/             GoogleTest correctness + precision suites
benchmarks/        Google Benchmark time and memory binaries
```

## Building

Requires CMake 3.16+ and a C++17 compiler. GoogleTest and Google Benchmark are
fetched automatically via `FetchContent`.

```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Useful options:

| Option                       | Default | Effect                                          |
|------------------------------|---------|-------------------------------------------------|
| `PSCOMP_BUILD_TESTS`         | `ON`    | Build the GoogleTest suites.                    |
| `PSCOMP_BUILD_BENCHMARKS`    | `ON`    | Build the Google Benchmark binaries.            |
| `PSCOMP_NATIVE_ARCH`         | `ON`    | Pass `-march=native` to enable auto-vectorize.  |

## Running the tests

```
ctest --test-dir build --output-on-failure
```

The suites covered:
* `test_known_series`           - closed-form regressions over `ModInt998`.
* `test_cross_algorithms_ntt`   - every variant must agree with the naive
  Horner reference on randomised NTT inputs (sizes 16..512).
* `test_random_ntt_large`       - `_opt` variants cross-check at sizes
  1024 and 2048 against `compose_brent_kung_opt`.
* `test_fft_cross_algorithm`    - pairwise FFT precision check on `double`
  and `long double`; reports `(min, q10, q25, q50, q75, q90, max, mean,
  var)` for every variant pair and asserts a per-`n` threshold.

## Running the benchmarks

```
./build/benchmarks/bench_compose_time   --benchmark_format=csv --benchmark_out=time.csv
./build/benchmarks/bench_compose_memory --benchmark_format=csv --benchmark_out=memory.csv
```

Time benchmarks cover every `(algorithm, optimization-level, backend, n)`
combination for `n in {256, 1024, 4096, 16384, 65536}`. The memory benchmark
reports both library-only allocation totals (via `pscomp::CountingResource`
plugged into the `std::pmr` default) and the process peak resident set size
(`getrusage`).

## API quick reference

```cpp
#include "pscomp/pscomp.hpp"
using namespace pscomp;

std::vector<ModInt998> f = ...;
std::vector<ModInt998> g = ...;            // requires g[0] == 0

ComposeOptions opts{ Algorithm::KinoshitaLi, OptLevel::Optimized };
auto h = compose<ModInt998>(f, g, /*n=*/4096, opts);
```

The dispatcher routes to the relevant template instantiation; for fine
control call the `algo::compose_*` entry points directly.

## CI

`.github/workflows/ci.yml` runs the GoogleTest suite on Ubuntu and macOS in
both Debug and Release configurations on every push and pull request.
