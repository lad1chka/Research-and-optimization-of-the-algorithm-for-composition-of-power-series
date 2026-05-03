# ALGORITHMS

This document specifies every algorithm shipped in `pscomp`, together with the
optimizations that distinguish each `_opt` variant from its `_basic` reference.
The narrative is meant to be read top-down: notation, finite-field arithmetic,
fast transforms, polynomial-level primitives, and finally the three families
of composition algorithms (naive, Brent-Kung, Kinoshita-Li).

For brevity we write `n` for the truncation length of every formal power
series (i.e. we operate in `R[[x]] / (x^n)`), `m` for `deg(f) + 1` (the length
of the outer series in a composition `f(g(x))`), and `R` for the coefficient
ring. Throughout, `g(0) = 0` is required so that `f(g(x)) mod x^n` is a well
defined truncated power series.

The proof of the Lagrange Inversion Theorem (used to derive certain
combinatorial identities, e.g. for the compositional inverse) is intentionally
omitted; we state it as a black box where needed.

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

### 2.3 Lazy reduction (described, not separately implemented)

For tight inner loops one can skip the post-addition normalisation as long as
intermediate values stay below `2P`. Since `2P < 2^31`, two consecutive
additions still fit in `uint32_t`, so a single normalisation can be deferred
until the next multiplication. We document this technique in the NTT section
but the code performs eager normalisation everywhere.

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

### 3.2 `fft::transform` (iterative + cache)

The optimized variant applies four standard improvements:

* **Iterative form.** A single in-place loop replaces the recursion stack:
  the input is first permuted by the bit-reversal of its index, then `log N`
  butterfly stages combine pairs `(a[i], a[i+half])`. No per-call allocations.
* **Cached bit-reverse table.** A `thread_local std::unordered_map<N, ...>`
  memoises the permutation; the second call at any size pays only `O(N)` for
  the swap loop.
* **Cached twiddles.** `omega_N^j` for `j = 0..N-1` are pre-tabulated per
  transform size and direction (forward / inverse). Eliminates trig calls
  from the hot path entirely.
* **Stride access pattern.** The butterfly uses `tw[j * step]` where
  `step = N / len`, so a single twiddle table per `N` covers every stage.

A scaling pass (`x *= 1/N`) is applied once on inverse transforms.

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

Floating-point FFTs accumulate rounding error of order `O(eps * log N)` per
transform with `eps = 2^{-52}` for `double`, `2^{-63}` for `x86 long double`.
Since composition repeatedly multiplies and truncates, the per-coefficient
error grows at least linearly with the depth of the algorithm. We do **not**
treat any FFT-backed routine as ground truth in tests; instead, we run every
algorithm on the same input and report deltas against the others (Section 9).

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

Recursive Cooley-Tukey, twiddles generated on the fly via repeated
multiplication by `omega_N`, multiplications go through `ModInt998Plain`
(i.e. through hardware division). Used as the *_basic* reference for every
NTT-backed benchmark.

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
Same asymptotic cost as `_def` but in practice slightly faster because every
multiplication is the same shape `(n) x (n) -> (n)` and the accumulator is
re-used in place. This is the pscomp reference used as the oracle in the
correctness tests.

### 6.3 `compose_naive_horner_inplace`

Same Horner loop but the accumulator is provided by the caller as an `out`
span and a re-usable scratch `vector` is passed in. Used as the leaf step
inside `compose_kl_recursion_threshold` so that small subproblems do not
each pay a `std::vector` allocation.

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

* **`compose_brent_kung_basic`.** Allocates `m_b + 1` power vectors and one
  vector per block. Uses `poly::mul_truncated` for every multiplication.
* **`compose_brent_kung_opt`.** Same algorithm, but the per-block accumulator
  is a single re-used buffer and the outer Horner runs in place. Removes
  `O(blocks)` `std::vector` allocations.
* **`compose_brent_kung_streaming`.** Builds `g^j` on the fly, one at a time;
  each `g^j` contributes to *all* blocks before being discarded. Memory drops
  from `O(sqrt(n) * n)` for the power table to `O(n)`. Time is unchanged.
* **`compose_brent_kung_tuned_m`.** Rounds `m_b` up to the next power of two
  so that the pad to the NTT transform length never wastes a stage. In our
  benchmarks this consistently shaves 5-20% off the runtime for large `n`.

### 7.3 Choice of `m_b`

The default is `ceil(sqrt(n))`. The tuned variant uses the next power of two
above `ceil(sqrt(n))`. Empirical sweeps (Section 9) confirm both heuristics
land within 10% of the optimum for `n` in `[2^8, 2^16]`.

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

We verified this recurrence directly on the closed-form example
`f = 1 + 2x + 3x^2`, `g = x + x^2`, `n = 4`, recovering
`f(g(x)) = 1 + 2x + 5x^2 + 6x^3` step by step (see `tests/correctness/
test_known_series.cpp`).

### 8.3 Cost analysis (this implementation)

The implementation in `compose_kl_*` truncates everything modulo `x^n` at
every step. The `y`-degree of `Q` stays `1`, the `y`-degree of `P` halves at
each step. Per-step bivariate multiplication of `P` (shape `n x m/2^i`) by
`Qm` (shape `n x 1`) costs `O(n * m / 2^i * log)`. Summed over the
`O(log m)` steps this gives an `O(n m polylog)` total -- on par with naive
in the worst case and slightly worse than Brent-Kung on average. The
`compose_kl_truncated_mul` variant additionally caps `deg_y(P)` at the
*current* target, halving the constant.

### 8.4 The optimal `O(n log^2 n)` complexity (Tellegen)

The asymptotically optimal bound from the Kinoshita-Li paper extracts the
*dual* coefficient `[x^{n-1}] rev(C)(x) / (1 - y g(x))`, where `C` is an
arbitrary length-`n` vector. Running bivariate Bostan-Mori in the `x`
direction shrinks `deg_x` by a factor of two and grows `deg_y(Q)` by a factor
of two per step, keeping the product `deg_x * deg_y` constant at `O(n)` and
yielding `O(n log^2 n)` total time.

That algorithm computes the linear map

```
T : (c_0, ..., c_{n-1}) |-> ([y^j] sum_k c_k [x^{n-1}] x^{n-1-k} g(x)^j)_{j}
                          = (sum_k c_k [x^k] g(x)^j)_{j}
                          = M^T c
```

where `M[i][j] = [x^i] g(x)^j` is the matrix of the composition map. By
Tellegen's principle, any algorithm computing `T = M^T` admits a *transposed*
algorithm of identical cost that computes `M` itself, i.e. composition.

The transposition is mechanical:
* every polynomial multiplication `X = A * B` (with `B` fixed) becomes a
  *middle product* `A := middleProduct(B, X)`,
* every "take even/odd in `x`" becomes "interleave even/odd into a single
  vector",
* every additive operation is unchanged (it is its own transpose).

Implementing the full transposed algorithm is a sizable engineering project
on its own and is left as future work; the present library implements the
identity directly (Section 8.2). When we cite "Kinoshita-Li" in benchmarks we
therefore mean the bivariate Bostan-Mori in the `y` direction, with the
caveat that the asymptotically optimal variant requires the Tellegen
transposition.

### 8.5 Variants

* **`compose_kl_basic`.** Reference implementation. Schoolbook `bi_mul`.
* **`compose_kl_inplace_bostan_mori`.** Re-uses three `BiPoly` buffers
  (`Qm`, `U`, `V`) across the loop body to drop the per-iteration allocations.
* **`compose_kl_arena_workspace`.** Threads a caller-owned `Arena` through the
  signature so a future zero-allocation rewrite can drop into place without
  changing callers; today it shares the body with `inplace`.
* **`compose_kl_truncated_mul`.** Specialises `bi_mul` for `deg_y(Q) = 1`
  (two truncated `x`-convolutions instead of a full `y`-convolution per
  `(i_1, i_2)` pair) and caps `deg_y(P)` to the current target after each
  step.
* **`compose_kl_packed_pq`.** Same algorithm with a flat row-major storage
  for `P` and `Q`. The current implementation aliases the truncated variant;
  the slot is reserved for a future `std::pmr::vector`-backed packed layout.
* **`compose_kl_recursion_threshold`.** When `n <= naive_threshold`, falls
  back to `compose_naive_horner_inplace`. Avoids the bivariate setup
  overhead for the leaf cases that dominate when used as a recursion base.

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

---

## 10. FFT precision testing methodology

Floating-point composition produces inevitable round-off error. The pscomp
test suite quantifies it as follows:

1. Generate one input pair `(f, g)` of random coefficients in `[-0.5, 0.5]`
   with `g(0) = 0`.
2. Run *every* composition entry point on this pair. None of them is treated
   as ground truth.
3. For every unordered pair of variants compute the per-coefficient
   absolute deltas; report the distribution as
   `(min, q10, q25, q50, q75, q90, max, mean, var)`.
4. Assert that the `max` delta is below an `n`-dependent threshold derived
   empirically from the typical `eps * log N` growth.

This isolates *relative* numerical disagreement between algorithms, which is
the right notion of "FFT error" for a composition library: the user does not
have access to the analytical answer and only cares about consistency
between fast paths.
