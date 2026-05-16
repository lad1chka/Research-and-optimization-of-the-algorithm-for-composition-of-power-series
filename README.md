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
[`docs/ALGORITHMS.md`](docs/ALGORITHMS.md).

## Repository layout

```
include/pscomp/
  field/           ModInt998 (Montgomery), ModInt998Plain, coef_traits
  transform/       fft, fft_basic, ntt, ntt_basic
  poly/            poly_mul, poly_utils
  compose/         naive, brent_kung, kinoshita_li, kinoshita_li_tellegen, api
  transpose/       middle_product, interleave, dual_extraction,
                   bi_convolution, flat_bipoly
  workspace/       Arena, CountingResource (std::pmr)
  span.hpp         C++17 substitute for std::span
  pscomp.hpp       umbrella header
src/dev_check.cpp  TU that pulls the umbrella header through warning flags
tests/             GoogleTest correctness and precision suites
benchmarks/        Google Benchmark time and memory binaries
scripts/           Precision capture and plotting scripts
docs/              Algorithm documentation (ALGORITHMS.md)
results/           Generated CSVs and figures (gitignored)
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
* `test_dual_extraction` — `extract_dual_*` variants checked against an
  `O(n^2 m)` oracle.
* `test_middle_product`, `test_compose_kl_tellegen`, `test_bi_convolution` —
  Tellegen pipeline and joint 2D-NTT primitives.

## Running the benchmarks

All benchmark CSVs and figures are collected in `results/`.

```bash
mkdir -p results

# Time benchmarks (all NTT-backed algorithms, n in {256, 1024, 4096, 16384, 65536})
./build/benchmarks/bench_compose_time \
    --benchmark_format=csv --benchmark_out=results/time.csv

# Memory benchmarks (library allocations + peak RSS)
./build/benchmarks/bench_compose_memory \
    --benchmark_format=csv --benchmark_out=results/memory.csv

# Precision (pairwise relative error between all FFT compose variants)
python scripts/capture_precision.py --output results/precision.csv
```

The memory benchmark overrides global `operator new`/`delete` to track peak
heap usage (`peak_heap_kB`) per algorithm run.

### Scripts

| Script | Description |
|--------|-------------|
| `scripts/capture_precision.py` | Runs `test_fft_cross_algorithm`, parses pairwise `rel_max` / `abs_max` from stdout and writes a CSV. Options: `--test-binary`, `--output`, `--filter`. |
| `scripts/plot_benchmarks.py` | Reads a CSV and generates heatmap PNGs. Three modes: `time` (Google Benchmark CSV → time heatmap), `memory` (memory CSV → peak arena heatmap), and `precision` (precision CSV → one error heatmap per `n`). Options: `--output-dir`. |

### Generating the figures

```bash
# Time heatmap
python scripts/plot_benchmarks.py time results/time.csv --output-dir results

# Memory heatmap (peak arena, MB)
python scripts/plot_benchmarks.py memory results/memory.csv --output-dir results

# Precision heatmaps (one per n)
python scripts/plot_benchmarks.py precision results/precision.csv --output-dir results
```

This writes `results/heatmap_time.png`, `results/heatmap_memory.png`,
and `results/heatmap_precision_n*.png`.

## API quick reference

```cpp
#include "pscomp/pscomp.hpp"
#include "pscomp/transpose/dual_extraction.hpp"
using namespace pscomp;

std::vector<ModInt998> f = ...;
std::vector<ModInt998> g = ...;            // g[0] must be 0
std::size_t n = 4096;

// Dispatcher API (selects algorithm and optimization level)
ComposeOptions opts;
opts.algo  = Algorithm::KinoshitaLi;
opts.level = OptLevel::Optimized;
auto h = compose<ModInt998>(f, g, n, opts);

// Direct call to the fastest variant — Kinoshita-Li dual, O(n log^2 n)
auto d = transpose::extract_dual_threshold<ModInt998>(f, g, n, n, 64);
```

The dispatcher routes to the relevant template instantiation; for fine
control call `algo::compose_*` or `transpose::extract_dual_*` entry points
directly.

## Composition complexities

| Algorithm | Time | Memory |
|-----------|------|--------|
| `compose_naive_horner` | `O(n^2)` | `O(n)` |
| `compose_brent_kung_basic` | `O(sqrt(n) M(n))` | `O(n^{1.5})` |
| `compose_brent_kung_opt` | `O(sqrt(n) M(n))` | `O(n^{1.5})` |
| `compose_brent_kung_streaming` | `O(sqrt(n) M(n))` | `O(n)` |
| `compose_brent_kung_tuned_m` | `O(sqrt(n) M(n))` | `O(n^{1.5})` |
| `compose_kl_truncated_mul` | `O(n^{1.5} polylog n)` | `O(n^{1.5})` |
| `compose_kl_recursion_threshold` | KL above threshold, naive below | as above |
| `extract_dual_basic` (KL dual) | `O(n log^2 n)` | `O(n log n)` |
| `extract_dual_inplace` (KL dual) | `O(n log^2 n)` | `O(n log n)` |
| **`extract_dual_truncated_mul`** (KL dual) | **`O(n log^2 n)`** | `O(n log n)` |
| `extract_dual_threshold` (KL dual) | brute fallback + `_truncated_mul` | `O(n log n)` |

`M(n) = O(n log n)` via NTT.  The `extract_dual_*` family implements the
Kinoshita-Li algorithm via x-axis Bostan-Mori with tight 2D NTT sizing (see
[`docs/ALGORITHMS.md` §8.6](docs/ALGORITHMS.md)).

## CI

`.github/workflows/ci.yml` runs the GoogleTest suite on Ubuntu and macOS,
Debug and Release.
