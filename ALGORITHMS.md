# ALGORITHMS

Specification of the algorithms shipped in `pscomp` and the optimizations
that distinguish the `_opt` variants from their `_basic` references. Sections
proceed from notation and finite-field arithmetic to fast transforms,
polynomial primitives, and the three composition families (naive,
Brent-Kung, Kinoshita-Li).

Throughout, `n` is the truncation length (operations are performed in
`R[[x]] / (x^n)`), `m = deg(f) + 1`, and `R` is the coefficient ring. We
require `g(0) = 0` so that `f(g(x)) mod x^n` is well defined.

The Lagrange Inversion Theorem is used as a black box where needed.

---

## 1. Notation and setup

A formal power series over `R` is written `f(x) = sum_{i>=0} a_i x^i`. A
truncated power series of length `n` is the equivalence class of `f` modulo
`x^n`; in code it is represented as a contiguous array of `n` coefficients.
Polynomial multiplication, inversion, exponentiation are all performed
modulo `x^n`. We refer to this as the *truncated* operation throughout.

Reversal: `rev_n(f)(x) = x^{n-1} f(1/x)`, equivalently the array of `f`
coefficients reversed. We use `rev(f)` for the reversal at the natural length.

Composition: `(f circ g)(x) = f(g(x))`. The condition `g(0) = 0` guarantees
that for every `j`, `g(x)^j` has order at least `j` in `x`, so the formal
expansion `f(g(x)) = sum_j a_j g(x)^j` has a well defined truncation modulo
`x^n` after summing only the terms with `j < n`.

Cost notation: `M(n)` denotes the cost of multiplying two polynomials of
degree `< n`. With FFT/NTT, `M(n) = O(n log n)`; with the schoolbook product,
`M(n) = O(n^2)`.

---

## 2. Modular arithmetic over `998244353`

We work over the NTT-friendly prime `P = 998244353 = 119 * 2^23 + 1`, which
admits a primitive root of order `2^23` (we use `g = 3`). All implementations
of NTT and `ModInt998` are restricted to transform sizes dividing `2^23`.

### 2.1 `ModInt998Plain`

The reference implementation stores values as `uint32_t` in the canonical
range `[0, P)`. Addition and subtraction normalise via a single conditional;
multiplication uses a 64-bit intermediate followed by a hardware division
(`% P`). Each multiplication therefore costs roughly 20-40 cycles on x86-64.

### 2.2 `ModInt998` (Montgomery form)

The optimized type stores values in *Montgomery form* `x_mont = x * R mod P`
with `R = 2^32`. All multiplications use the Montgomery reduction

```
reduce(a) = ((a + ((a mod R) * (-P^{-1} mod R) mod R) * P) >> 32) mod P
```

which replaces the hardware division by a single 32-bit multiplication and a
`64 -> 32`-bit shift, dropping the per-mul cost to roughly 5-7 cycles.

The constants `R mod P`, `R^2 mod P` and `-P^{-1} mod 2^32` are computed at
compile time (Newton iteration for the inverse, doubling correct bits each
step).

### 2.3 Lazy reduction (described, not implemented)

In tight inner loops the post-addition normalisation can be deferred while
intermediate values stay below `2P`. Since `2P < 2^31`, two consecutive
additions still fit in `uint32_t`. The current code performs eager
normalisation everywhere.

---

## 3. Fast Fourier Transform (FFT)

The FFT computes the discrete Fourier transform of a length-`N` vector with
`N = 2^k` in `O(N log N)` operations using the Cooley-Tukey decimation-in-
time recurrence

```
DFT_N(a)[j] = sum_{i=0}^{N-1} a_i omega_N^{ij}
            = DFT_{N/2}(a_even)[j] + omega_N^j * DFT_{N/2}(a_odd)[j].
```

### 3.1 `fft_basic::transform` (recursive reference)

The reference implementation realises the recurrence verbatim: at each level
it allocates two `N/2`-vectors (`evens`, `odds`), recurses, and combines with
twiddle factors computed on the fly. Total allocations: `O(N)` across all
recursion levels; total trig calls: `O(N log N)`.

### 3.2 `fft::transform` (iterative, cached)

Four standard improvements over §3.1:

* Iterative form. A single in-place loop replaces the recursion: bit-reverse
  permute, then `log N` butterfly stages combine pairs `(a[i], a[i+half])`.
  No per-call allocations.
* Cached bit-reverse table per `N` (`thread_local`).
* Cached twiddles `omega_N^j` per transform size and direction.
* Stride access `tw[j * step]` with `step = N / len` covers every stage.

The inverse transform applies `x *= 1/N` once at the end.

### 3.3 Real-valued packed transform

`fft::packed_real_forward(a, b, ...)` packs two real-valued inputs `a` and
`b` into the real and imaginary parts of a single complex array, executes one
forward FFT, and recovers the spectra of `a` and `b` from the symmetry

```
A[k] = (Z[k] + conj(Z[N-k])) / 2,
B[k] = (Z[k] - conj(Z[N-k])) / (2 i).
```

This halves the FFT count when convolving two real sequences (e.g. inside
`fft::multiply`).

### 3.4 Numerical error

Each forward/inverse pair accumulates rounding of order `O(eps * log N)` with
`eps = 2^{-52}` for `double` and `2^{-63}` for x86 `long double`. Composition
multiplies and truncates repeatedly, so the per-coefficient error grows at
least linearly with the algorithmic depth. No FFT-backed routine is treated
as ground truth; tests run every variant on the same input and report deltas
against the others (§10).

---

## 4. Number-Theoretic Transform (NTT)

The NTT is the FFT with the complex root of unity `omega_N = e^{2 pi i / N}`
replaced by an order-`N` element of `(Z/PZ)^*`. We use

```
omega_N = g^{(P - 1) / N} mod P,    g = 3.
```

Since `P - 1` is divisible by `2^23`, every `N | 2^23` admits such an
`omega_N`; this caps the NTT-supported truncation length at `2^23`.

### 4.1 `ntt_basic::transform` (reference)

Recursive Cooley-Tukey with on-the-fly twiddles and `ModInt998Plain`
arithmetic (hardware division). Used as the `_basic` reference for NTT
benchmarks.

### 4.2 `ntt::transform` (optimized)

Combines the same four optimizations as `fft::transform` plus the Montgomery
arithmetic of `ModInt998`. The inner butterfly becomes

```
u = a[base + j];
v = a[base + j + half] * tw[j * step];      // single Montgomery mul
a[base + j]        = u + v;                  // mod-P add
a[base + j + half] = u - v;                  // mod-P sub
```

so each butterfly costs one Montgomery multiplication and two conditional
subtractions.

### 4.3 Convolution

`ntt::multiply(a, b)` zero-pads both inputs to the next power of two `>=
|a| + |b| - 1`, runs forward NTT on each, multiplies pointwise, then runs an
inverse NTT and trims trailing zeros. This is the workhorse called by every
`poly::mul_truncated` invocation in the optimized backends.

---

## 5. Polynomial layer

`poly::mul_schoolbook` implements the textbook `O(n m)` product and is used
exclusively as a baseline. `poly::mul_fft` and `poly::mul_ntt` wrap their
respective transform's convolution. `poly::mul` dispatches at compile time:

```
ModInt998       -> mul_ntt        (Montgomery NTT)
ModInt998Plain  -> mul_ntt_basic  (recursive NTT, plain `% P`)
double / long double -> mul_fft   (iterative complex FFT)
otherwise       -> mul_schoolbook
```

`poly::mul_truncated(a, b, n)` computes the full product and truncates the
result to length `n`. The KL truncated-mul variant exploits this to avoid
producing coefficients past the final truncation length even during recursion.

`poly::resized`, `poly::reversed` and `poly::round_up_pow2` are the small
helpers shared across composition implementations.

---

## 6. Composition: naive baselines

### 6.1 `compose_naive_def`

Computes `g^0, g^1, ..., g^{m-1}` modulo `x^n` and accumulates
`sum_j a_j g^j` term by term. Uses `m - 1` truncated polynomial
multiplications and `m` length-`n` linear combinations. Total cost
`O(m * (M(n) + n))`. Memory: `O(n)` for the running power and `O(n)` for the
accumulator.

### 6.2 `compose_naive_horner`

Horner scheme:
```
result := a_{m-1}
for k = m - 2 down to 0:
    result := (result * g) mod x^n + a_k
```
Same asymptotic cost as `_def`. Every multiplication has the same shape
`(n) x (n) -> (n)` and reuses the accumulator. Serves as the oracle in the
correctness tests.

### 6.3 `compose_naive_horner_inplace`

Same loop with caller-owned `out` span and scratch vector. Used as the leaf
step inside `compose_kl_recursion_threshold` to avoid per-call allocations.

---

## 7. Composition: Brent-Kung (1978)

### 7.1 Idea

Pick a block size `m_b ~ sqrt(m)`. Split `f(y)` into blocks of `m_b`
coefficients each:

```
f(y) = F_0(y) + F_1(y) y^{m_b} + F_2(y) y^{2 m_b} + ...,    deg F_k < m_b.
```

Then

```
f(g(x)) = sum_k F_k(g(x)) * (g(x)^{m_b})^k.
```

Two stages:

1. **Power table.** Compute `g(x)^0, g(x)^1, ..., g(x)^{m_b}` modulo `x^n`.
   Cost: `m_b - 1` truncated multiplications, `O(m_b * M(n))`.
2. **Block evaluation.** For each block `k`, accumulate
   `F_k(g(x)) = sum_{j < m_b} f[k * m_b + j] * g^j(x)` as a single
   length-`n` linear combination. Cost: `O(m_b * n)` per block.
3. **Outer Horner.** Combine the blocks by Horner in `g^{m_b}`:
   `result = ((... (F_{K-1} g^{m_b} + F_{K-2}) g^{m_b} + ...) g^{m_b} + F_0`.
   Cost: `K - 1 = O(m / m_b)` truncated multiplications, `O(m / m_b * M(n))`.

Total cost: `O((sqrt(m) + m / sqrt(m)) * M(n) + m * n) = O(sqrt(m) * M(n) +
m * n)`. With `m = n` and `M(n) = n log n` this is `O(n^{1.5} log n + n^2)`,
asymptotically `O(n^2)` but with significantly fewer FFTs/NTTs than the
naive `O(n)` scheme.

### 7.2 Variants

* `compose_brent_kung_basic` — allocates `m_b + 1` power vectors and one
  vector per block; uses `poly::mul_truncated`.
* `compose_brent_kung_opt` — single reused per-block accumulator, outer
  Horner runs in place.
* `compose_brent_kung_streaming` — builds `g^j` one at a time; each `g^j`
  contributes to all blocks before being discarded. Memory `O(n)` instead
  of `O(sqrt(n) n)`; time unchanged.
* `compose_brent_kung_tuned_m` — rounds `m_b` up to the next power of two so
  the NTT pad does not waste a stage.

### 7.3 Choice of `m_b`

Default `ceil(sqrt(n))`; tuned variant uses the next power of two above
`ceil(sqrt(n))`. Empirical sweeps (§10) place both heuristics within 10% of
the optimum for `n in [2^8, 2^16]`.

---

## 8. Composition: Kinoshita-Li (2024)

### 8.1 The bivariate identity

Starting from the geometric expansion `1 / (1 - y g(x)) = sum_{j >= 0} y^j
g(x)^j`, observe that

```
[y^{m-1}] rev(f)(y) / (1 - y g(x))
  = sum_{j+k=m-1} f_{m-1-j} g(x)^k
  = sum_{k=0}^{m-1} f_k g(x)^k
  = f(g(x)).
```

Hence `f(g(x)) mod x^n` equals the `y^{m-1}` coefficient of the bivariate
rational function `rev(f)(y) / (1 - y g(x))`, viewed as a power series in `x`
truncated mod `x^n`.

### 8.2 Bivariate Bostan-Mori recursion

To extract one coefficient of a univariate rational function, the classical
Bostan-Mori recurrence proceeds as follows: to extract `[x^k] P(x) / Q(x)`,
multiply numerator and denominator by `Q(-x)`. Then `Q(x) Q(-x)` contains
only even powers of `x`, so the result is `[x^k] (P(x) Q(-x)) / V(x^2)` with
`V(z)` half the size of the original denominator. Splitting `P(x) Q(-x)` into
even and odd `x`-parts halves the target index `k`.

The bivariate version of the same idea operates on polynomials in `x` whose
coefficients are themselves polynomials in `y`. Applied to the `y` direction
of `rev(f)(y) / (1 - y g(x))` it gives:

```
function bm_y(P, Q, k, n):
    while k > 0:
        Qm := Q with sign of every odd-y term flipped  // y -> -y
        U  := (P * Qm) mod x^n                          // bivariate multiply
        V  := (Q * Qm) mod x^n                          // even in y by construction
        if k odd:  P := y-odd-part(U);  k := (k - 1) / 2
        else:      P := y-even-part(U); k := k / 2
        Q := y-even-part(V) compressed to a poly in z := y^2
    return P[i][0] for i = 0..n-1     // q0 = (1, 0, 0, ...) by induction
```

Initialisation: `P[0] = rev(f)`, `P[i] = 0` for `i > 0`; `Q[0] = (1, 0)`,
`Q[i] = (0, -g[i])` for `i >= 1`. The initial `y`-target is `m - 1`.

The recurrence is checked on `f = 1 + 2x + 3x^2`, `g = x + x^2`, `n = 4`,
recovering `f(g(x)) = 1 + 2x + 5x^2 + 6x^3`
(see `tests/correctness/test_known_series.cpp`).

### 8.3 Cost (current implementation)

Every step truncates modulo `x^n`. `deg_y(Q) = 1` is preserved; `deg_y(P)`
halves. Per step the bivariate multiplication of `P` (shape `n x m/2^i`) by
`Qm` (shape `n x 1`) costs `O(n * m / 2^i * log)`. Summed over `O(log m)`
steps the total is `O(n m polylog)`. `compose_kl_truncated_mul` additionally
caps `deg_y(P)` at the current target.

### 8.4 Dual problem and Tellegen transposition

For `c \in R^n` define

```
d_j = sum_{k=0..n-1} c_k * [x^k] g(x)^j,   j = 0..m-1.
```

`d = M^T c`, where `M[i][j] = [x^i] g(x)^j` is the matrix of the composition
map `f -> f(g)`. By Tellegen's principle, an algorithm computing `M^T c`
admits a transposed algorithm of identical cost that computes `M f`.

#### 8.4.1 Forward dual (`extract_dual`)

The identity `sum_k c_k [x^k] A(x) = [x^{n-1}] rev_n(c)(x) * A(x)` rewrites
the dual sum as

```
d(y) = [x^{n-1}] rev_n(c)(x) / (1 - y g(x))   mod y^m.
```

Bostan-Mori in `x` extracts the `[x^{n-1}]` coefficient. Each step

* multiplies numerator and denominator by `Q(-x, y)` (`negate_odd_x`);
* contracts the x-axis by 2 (`take_x_parity`).

After `O(log n)` outer iterations only `d(y)` remains. Implementation:
[include/pscomp/transpose/dual_extraction.hpp](include/pscomp/transpose/dual_extraction.hpp);
validated against an `O(n^2 m)` oracle in
[tests/correctness/test_dual_extraction.cpp](tests/correctness/test_dual_extraction.cpp).

#### 8.4.2 Transposed sweep (`compose_kl_tellegen`)

The forward sweep is decomposed into elementary linear operations whose
transposes are listed in
[include/pscomp/transpose/DAG_FORWARD_DUAL.md](include/pscomp/transpose/DAG_FORWARD_DUAL.md):

| forward operation                | transpose                                   |
|----------------------------------|---------------------------------------------|
| `take_x_parity(U, p)`            | `interleave_x_parity(P', p)`                |
| `bi_mul_x(P, Qm) mod x^{x_cap}`  | `bi_middle_product_x(Qm, U_t, x_cap)`       |
| `negate_odd_x(Q)`                | unchanged                                   |
| pack `rev_n(c)` into `P[*][0]`   | read `P[*][0]`, reverse                     |

`compose_kl_tellegen<Coef>(f, g, n)`
([include/pscomp/compose/kinoshita_li_tellegen.hpp](include/pscomp/compose/kinoshita_li_tellegen.hpp))
replays the forward Q sweep via `sweep_forward_g`, walks the level metadata
in reverse, and at each level applies `interleave_x_parity` followed by
`bi_middle_product_x` against the precomputed `Qm`. The result is read from
the `y^0` column of the final bivariate buffer and reversed.

Bit-for-bit agreement with `compose_kl_basic` is checked in
[tests/correctness/test_compose_kl_tellegen.cpp](tests/correctness/test_compose_kl_tellegen.cpp)
on `g(x) = x`, `f = e_0`, `f = e_1`, and random inputs up to `n = 64`. The
variant is also part of the cross-algorithm precision suite via
`tests/test_helpers.hpp::all_entries`.

#### 8.4.3 Cost (current implementation)

`bi_middle_product_x` runs one univariate middle product per `(jB, jU)`
y-pair. Total work is `O(n m polylog n)`: each level halves `x_cap`, while
the Q-tower carries y-extent up to `m`. Reaching `O(n log^2 n)` requires
either (i) batching all y-pairs into one bivariate transform, or (ii) the
y-axis Bostan-Mori contraction from the original paper. See §8.6.

#### 8.4.4 API

```cpp
pscomp::ComposeOptions opts;
opts.algo       = pscomp::Algorithm::KinoshitaLi;
opts.kl_variant = pscomp::KLVariant::Tellegen;
auto out = pscomp::compose<pscomp::ModInt998>(f, g, n, opts);
```

Default `KLVariant::Forward` keeps the forward Bostan-Mori path.

### 8.5 Variants

* `compose_kl_basic` — reference implementation; schoolbook `bi_mul`.
* `compose_kl_inplace_bostan_mori` — reuses `Qm`, `U`, `V` across the loop.
* `compose_kl_arena_workspace` — accepts a caller-owned `Arena` (currently
  shares the body with `inplace`).
* `compose_kl_truncated_mul` — specialises `bi_mul` for `deg_y(Q) = 1` (two
  truncated x-convolutions per `(i_1, i_2)` pair) and caps `deg_y(P)` at
  the current target.
* `compose_kl_packed_pq` — placeholder for a flat row-major `P, Q` layout;
  currently aliases `_truncated_mul`.
* `compose_kl_recursion_threshold` — falls back to
  `compose_naive_horner_inplace` for `n <= naive_threshold`.
* `compose_kl_tellegen` — Tellegen-transposed dual sweep (§8.4). Selected
  via `KLVariant::Tellegen`. Output bit-for-bit identical to `compose_kl_basic`;
  current cost `O(n m polylog n)`.

### 8.6 Future work toward `O(n log^2 n)`

Two gaps separate the current `compose_kl_tellegen` from the Kinoshita-Li
bound:

1. Joint y-axis convolution. `bi_mul_x_truncated` and `bi_middle_product_x`
   run one univariate transform per y-pair. A single bivariate FFT/NTT
   (e.g. packing y-rows into a length-`2m` FFT) amortises twiddle and
   bit-reversal work.
2. y-axis Bostan-Mori contraction. The original paper pairs the x-axis
   recursion with a y-axis recursion that keeps `deg_y(Q) = O(1)` after
   each x-step, restoring `O(n log^2 n)`. This requires a second DAG and
   its transpose mirroring §8.4 along y.

Until both are implemented, `compose_kl_tellegen` is a correctness milestone:
it shows that the transposed pipeline (middle product, interleave, level
replay) reproduces the forward output exactly. Performance crossover with
`kl_threshold` lies beyond the sizes the current bivariate convolution can
afford.

---

## 9. Optimization summary

| Layer | `_basic`                                | `_opt`                                                                                  |
|-------|-----------------------------------------|-----------------------------------------------------------------------------------------|
| ModP  | `% P` per multiplication                | Montgomery form, single 32-bit mul                                                      |
| FFT   | recursive, on-the-fly twiddles, `O(N)` allocs | iterative in-place, cached twiddles + bit-reverse, packed real trick                    |
| NTT   | recursive, plain `% P`                  | iterative in-place, cached twiddles + bit-reverse, Montgomery, lazy add (described)     |
| Mul   | schoolbook                              | NTT/FFT convolution                                                                     |
| Naive | `_def`                                  | `_horner` (in-place accumulator), `_horner_inplace` (caller-owned workspace)            |
| BK    | `_basic`                                | `_opt` (single accumulator), `_streaming` (O(n) memory), `_tuned_m` (power-of-two `m_b`) |
| KL    | `_basic` (schoolbook bi-mul)            | `_inplace_bostan_mori` (buffer reuse), `_truncated_mul` (deg_y(Q)=1 specialisation),    |
|       |                                         | `_arena_workspace` (caller arena), `_packed_pq` (flat layout), `_recursion_threshold`   |
| KL-T  | -                                       | `_tellegen` (DAG transposition via middle product / interleave; §8.4)                   |

---

## 10. FFT precision testing methodology

Floating-point composition produces round-off error. The pscomp test suite
quantifies it as follows:

1. Generate one input pair `(f, g)` of random coefficients in `[-0.5, 0.5]`
   with `g(0) = 0`.
2. Run every composition entry point on the pair. None is treated as ground
   truth.
3. For each unordered pair compute per-coefficient absolute and relative
   deltas; report `(min, q10, q25, q50, q75, q90, max, mean, var)`.
4. Assert the relative max is below an `n`-dependent threshold derived from
   the `eps * log N` growth.

The test isolates pairwise numerical disagreement between algorithms. No
analytical answer is available in the general case; the relevant property
is consistency across the fast paths.
