# MTFP21: Multi-Trit Floating Point (21-Trit)

## Overview

MTFP21 is a 21-trit floating point representation using balanced ternary {-1, 0, +1}. It exceeds IEEE 754 float32 in both mantissa precision and dynamic range while executing entirely in the AVX2 integer pipeline via `sign_epi8`.

**Blackjack.** 21 is where precision, range, and hardware alignment converge.

## The Split: 16 + 5

```
┌────────────────────────────────────────────────────────┐
│                      MTFP21                             │
│                                                         │
│  ┌──────────────────────────────┐  ┌─────────────────┐  │
│  │     16-trit mantissa         │  │  5-trit exponent │  │
│  │     balanced ternary         │  │  balanced ternary│  │
│  │     sign is implicit         │  │  base-3 scaling  │  │
│  └──────────────────────────────┘  └─────────────────┘  │
│                                                         │
│  Total information: 33.3 bits                           │
│  Storage: 21 bytes (byte-per-trit) or 6 bytes (packed)  │
└────────────────────────────────────────────────────────┘
```

## MTFP21 vs IEEE 754 Float32

| Property | Float32 (IEEE 754) | MTFP21 |
|----------|-------------------|--------|
| Total bits/trits | 32 bits | 21 trits |
| Total information | 32 bits | 33.3 bits |
| Sign | 1 bit (dedicated) | Free — implicit in balanced ternary |
| Mantissa | 23 bits + 1 implicit = 24 bits | 16 trits = 25.4 bits |
| Mantissa states | 16,777,216 | 43,046,721 |
| Decimal precision | ~7.2 digits | ~7.6 digits |
| Exponent | 8 bits = 256 levels | 5 trits = 243 levels |
| Dynamic range | 2^-126 to 2^+127 ≈ 10^±38 | 3^-121 to 3^+121 ≈ 10^±57 |
| Multiply | FP multiplier circuit | `sign_epi8` — 1 cycle, 32-wide integer |
| Add | FP adder with alignment | `add_epi8` after exponent alignment |
| Sign overhead | 1 bit per value | Zero — balanced ternary IS signed |
| Sparsity | None — every bit carries signal | Native — trit 0 = "don't care" |
| Hardware | FPU / AVX float pipeline | **AVX2 integer pipeline** |

MTFP21 exceeds float32 in:
- Mantissa precision (25.4 vs 24 bits — 2.57× more states)
- Dynamic range (10^±57 vs 10^±38)
- Information density (33.3 bits in 21 trits vs 32 bits in 32 bits)
- Arithmetic cost (integer sign_epi8 vs floating-point multiply)

Float32 wastes 1 of 32 bits on a sign bit. MTFP21 wastes nothing — sign is structural.

## AVX2 Register Layout

### Byte-per-trit (compute format)

Each trit stored as one int8 value {-1, 0, +1}. This is the format used for arithmetic — `sign_epi8` operates directly on these lanes.

**Mantissa:**
```
AVX2 register (32 × int8):
┌────────────────────────────┬────────────────────────────┐
│  Mantissa A (16 × int8)    │  Mantissa B (16 × int8)    │
│  trits [0..15]             │  trits [0..15]             │
└────────────────────────────┴────────────────────────────┘
Two MTFP21 mantissas per register. Zero waste.
```

**Exponent:**
```
Stored in a parallel int8 array.
5 trits per exponent, packed or byte-per-trit.
Exponent arithmetic: add_epi8 (multiply), max_epi8 (align for add).
```

### Packed (storage format)

For memory efficiency, 21 trits can be packed at 2 bits per trit:

- 21 trits × 2 bits = 42 bits = 5.25 bytes
- Pad to 6 bytes (48 bits, 3 trits of padding)
- Or pack 4 MTFP21 values into 21 bytes (no padding waste)

Packing and unpacking are SIMD-friendly — shift, mask, and sign-extend operations.

## Mantissa: 16 Trits

The mantissa is a 16-trit balanced ternary value. Each trit is weighted by its position:

```
value = Σ(i=0 to 15) trit[i] × 3^i
```

**Range:** -(3^16 - 1)/2 to +(3^16 - 1)/2 = -21,523,360 to +21,523,360

**Precision:** 43,046,721 distinct values ≈ 25.4 bits

**Key property:** Sign is encoded in the trits themselves. A positive value has more +1 trits; a negative value has more -1 trits. Negation is `sign_epi8(mantissa, all_minus_ones)` — flip every trit in one cycle.

**Sparsity:** A mantissa with many zero trits is simultaneously:
- A valid number (the zeros don't contribute to the value)
- A sparse representation (fewer active trits = less computation for routing)
- A signal about precision requirements (sparse mantissa = coarse value = maybe doesn't need full precision)

## Exponent: 5 Trits

The exponent is a 5-trit balanced ternary value representing a power of 3:

```
scale = 3^exponent
exponent = Σ(i=0 to 4) trit[i] × 3^i
```

**Range:** -121 to +121

**Full value:** `value = mantissa × 3^exponent`

**Dynamic range:** 3^-121 to 3^+121 ≈ 10^-57.7 to 10^+57.7

### Why base 3

The exponent uses base 3, not base 2 or base 10. This is natural — the mantissa is ternary, the exponent is ternary, the scaling is ternary. The entire number lives in one number system.

Base-3 exponent also means that multiplying by 3 (a common scaling operation in ternary systems) is just incrementing the exponent by 1 — a single trit operation. In float32, multiplying by 3 requires an actual floating-point multiply.

## Arithmetic

### Multiply

```
result.mantissa = sign_epi8(a.mantissa, b.mantissa)   // 1 cycle, 16-wide
result.exponent = add_epi8(a.exponent, b.exponent)     // 1 cycle
```

Two cycles. Integer pipeline. No floating-point unit involved.

For a dot product (the fundamental operation of neural inference):

```c
// MTFP21 dot product of two 16-trit mantissa vectors
// (assuming aligned exponents — see normalization)
__m256i va = _mm256_load_si256((__m256i *)a_mantissa);  // 2 mantissas
__m256i vb = _mm256_load_si256((__m256i *)b_mantissa);  // 2 mantissas
__m256i prod = _mm256_sign_epi8(va, vb);                // ternary multiply
__m256i acc = _mm256_add_epi8(acc, prod);               // accumulate
// exponent of result = sum of input exponents
```

### Add

Addition requires exponent alignment — same as float32, but in balanced ternary:

```
1. Compare exponents: diff = a.exponent - b.exponent
2. Shift the smaller mantissa right by |diff| positions
   (right-shift in balanced ternary = divide by 3^|diff|)
3. Add mantissas: add_epi8(a.mantissa, shifted_b.mantissa)
4. Result exponent = max(a.exponent, b.exponent)
5. Normalize: if mantissa overflows, shift right and increment exponent
```

Ternary right-shift is trit-wise: drop the least significant trit, shift remaining trits down one position. SIMD-friendly via `_mm256_alignr_epi8` or byte shuffle.

### Negate

```
result.mantissa = sign_epi8(a.mantissa, all_minus_ones)  // 1 cycle
result.exponent = a.exponent                              // unchanged
```

One cycle. Flip every trit. No special cases, no two's complement conversion, no sign bit manipulation.

### Compare

```
1. Compare exponents first (higher exponent = larger magnitude)
2. If equal, compare mantissas trit-by-trit from most significant
```

### Absolute Value

```
result.mantissa = _mm256_abs_epi8(a.mantissa)  // 1 cycle
result.exponent = a.exponent
```

## Normalization

A normalized MTFP21 has its most significant trit non-zero (either -1 or +1). This maximizes precision — no leading zeros wasting mantissa resolution.

Normalization after arithmetic:
1. Count leading zero trits in mantissa
2. Shift mantissa left by that count (multiply by 3^count)
3. Subtract count from exponent

This is analogous to float32 normalization but in balanced ternary.

## Special Values

| Value | Mantissa | Exponent |
|-------|----------|----------|
| Zero | all zeros | any (conventionally all zeros) |
| +1 | [0,0,...,0,+1] | [0,0,0,0,0] |
| -1 | [0,0,...,0,-1] | [0,0,0,0,0] |
| +3 | [0,0,...,0,+1] | [0,0,0,0,+1] |
| Max positive | [+1,+1,...,+1] | [+1,+1,+1,+1,+1] |
| Max negative | [-1,-1,...,-1] | [+1,+1,+1,+1,+1] |
| Smallest nonzero | [0,0,...,0,±1] | [-1,-1,-1,-1,-1] |

No NaN. No infinity. No denormals. No negative zero. Balanced ternary doesn't have these edge cases — every trit pattern is a valid number. This eliminates an entire class of floating-point bugs.

## Connecting to the Six Primes

MTFP21 is the representation that lets all six primes execute on the CPU integer pipeline:

| Prime | MTFP21 implementation | Pipeline |
|-------|----------------------|----------|
| ADD | Exponent align + `add_epi8` | CPU integer |
| MUL | `sign_epi8` mantissa + `add_epi8` exponent | CPU integer |
| MAX | Exponent compare, then mantissa compare | CPU integer |
| CONST | Fixed trit pattern | CPU immediate |
| EXP | Frozen shape: ternary lookup table approximation | CPU integer |
| LOG | Frozen shape: ternary lookup table approximation | CPU integer |

If EXP and LOG can be approximated with sufficient precision using frozen shapes (ternary lookup tables composed from the other four primes), then all six primes stay in the integer pipeline. The iGPU becomes a high-precision fallback, not a requirement.

The precision threshold: can a frozen shape approximate `exp(x)` and `log(x)` to within MTFP21's 25.4-bit mantissa resolution using ternary lookup + interpolation? If yes, the entire compute fabric runs on one substrate. If no, the transcendentals dispatch to the iGPU as originally designed — still a win, just not full unification.

## Implementation Status

MTFP21 is **fully implemented and validated** in `bitnet/shirley_mtfp21.h` (109/109 tests pass):

- Conversion: float32 ↔ MTFP21, int8 ↔ MTFP21 (exact round-trip for all trit ranges)
- Arithmetic: add (with exponent alignment), multiply (single-shot normalization), negate, abs
- Division: integer reciprocal multiply (no float), scalar and general variants
- rsqrt: 256-entry precomputed LUT + 2 Newton-Raphson iterations, zero float ops, max error 8.94e-08
- **exp(x):** 256-entry LUT + linear interpolation. Range reduction via x = k×ln(3) + r exploits MTFP21's base-3 exponent — 3^k is a free exponent shift. Max error 9.50e-06 across softmax range [-30, 0]. Handles exp(70) ≈ 2.5e30 correctly. (commit 6bb5ccb)
- **softmax:** Full MTFP21: subtract max (MAX prime) → exp (EXP prime) → sum (ADD prime) → normalize via reciprocal (MUL prime) → quantize to int8. Conservation validated (sum ≈ out_range). (commit 6bb5ccb)
- **cmp(a, b):** Integer comparison without float conversion. Returns -1/0/+1. Used for softmax max-finding. (commit 6bb5ccb)
- RMSNorm: pure MTFP21 path (zero float between conversion boundaries), 100% exact match vs float64
- Accumulation: statistically better than float32 (wins 57.2% of 1000 head-to-head comparisons)
- Adversarial: passes all 6 distributions (heavy tails, near-zero, bimodal, cancellation, dynamic range, all-same)

### Correction: Multiply spec

The original spec described multiply as `sign_epi8(a.mantissa, b.mantissa)` — element-wise trit multiplication. This is only correct when one operand is a single ternary weight {-1, 0, +1} (the matmul case). General MTFP21 × MTFP21 uses int32 mantissa multiplication in int64 with single-shot normalization. The implementation uses the correct general algorithm.

## Architectural Role

MTFP21 serves two roles in the Shirley pipeline:

**1. Full-precision compute between matmuls.** The FFN block (commit b0489e1) runs every intermediate value — gate output, up output, ReLU, square, element-wise multiply — in MTFP21 at 25.4-bit precision. No quantization between operations. The per-element MTFP21 arithmetic (scalar int64 loops) is slower than AVX2 (~7 tok/s vs ~22 tok/s) but preserves precision: PPL matches baseline within noise, while int8 intermediates lost 1+ PPL.

**2. Scale factor bookkeeping.** Every int8 tensor at the matmul interface carries an MTFP21 scale. Every matmul rescale computes `weight_scale × layer_scale / act_scale` in MTFP21. These are scalar operations — the overhead is negligible.

The performance gap (scalar MTFP21 vs AVX2 vectorized) is an optimization target, not an architectural problem. The math is correct at full precision. AVX2 vectorization of MTFP21 operations (int32 mantissa in SIMD lanes, exponent tracking) is future work.

```
Layer 1: AVX2 integer kernels    Matmul inner loop (sign_epi8, maddubs)
         Ternary dot product     32 elements/cycle, native hardware

Layer 2: MTFP21 per-element      Between-matmul operations (ReLU, sqr, mul, norm)
         Full precision          25.4 bits, no quantization boundaries

Layer 3: MTFP21 scalar           Scale factors, rsqrt, exp (one-off per call)
         LUT + Newton-Raphson    Integer-only transcendentals
```

MTFP21 sits at Layer 3. It stores the scalar results of Layer 2 computations (rsqrt values, reciprocals, scale factors) and transports them between Layer 1 compute stages. It does NOT perform the per-element squaring, accumulation, or scaling — those use raw AVX2 integer at Layer 1.

The `shirley_rmsnorm_ternary` kernel demonstrates this architecture:
- Phase 1+2: AVX2 int8→int16→int32 sum of squares (Layer 1)
- Phase 3: MTFP21 integer rsqrt via LUT+NR (Layer 2/3)
- Phase 4: MTFP21→Q15 fixed-point conversion (Layer 3→1 bridge)
- Phase 5: AVX2 `_mm256_mulhrs_epi16` scale + pack (Layer 1)
- Result: 417 ns, zero float, 100% exact, faster than the float path

## Origin

Conceived March 31, 2026 during Shirley project design. Implemented April 1, 2026. Repositioned from compute format to transport format after profiling revealed 65x overhead vs float32 for per-element operations. The proof of concept (102/102 tests, better-than-float32 accumulation) validates the number system; the three-layer architecture puts it where it belongs.
