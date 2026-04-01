# SYNTHESIZE: AVX2 Integer RMSNorm Kernel

## The Design in One Sentence

Replace the 180 us MTFP21 RMSNorm with a 50-line AVX2 integer kernel that squares and accumulates in int16/int32 via native SIMD, takes one float rsqrt, and scales via fixed-point multiply — targeting < 1 us per call.

## Architecture: Three-Layer Compute Stack

```
Layer 3: MTFP21                  Transport/storage format
         ─────────────────────── ternary values with dynamic range
         Weight scales, norm params, metadata between stages

Layer 2: Scalar arithmetic       One-off computations  
         ─────────────────────── FPU for rsqrt, iGPU for batched EXP/LOG
         Called once per RMSNorm, not per element

Layer 1: AVX2 integer kernels    Per-element bulk compute
         ─────────────────────── sign_epi8, mullo_epi16, madd_epi16
         32 elements per cycle, native hardware
```

The kernel lives at Layer 1. The rsqrt is Layer 2. MTFP21 is Layer 3. Each layer uses the substrate it was designed for.

## Kernel Specification

```c
/*
 * ternary_rmsnorm_avx2 — Integer RMSNorm for ternary inference pipeline
 *
 * Input:  int8 activations from matmul (range ±SHIRLEY_ACT_RANGE)
 * Output: int8 normalized activations (range ±out_range)
 *
 * Compute: y[i] = round(x[i] * rsqrt(mean(x^2) + eps) * out_range_scale)
 *
 * All per-element work in AVX2 integer. One float rsqrt.
 */
void ternary_rmsnorm_avx2(
    int8_t       *dst,        // output: normalized int8
    const int8_t *src,        // input: int8 activations
    int           n,          // dimension (e.g. 2560)
    float         eps,        // epsilon (e.g. 1e-5)
    int           out_range   // output clamp (e.g. 80 for 5-trit)
);
```

### Phase 1+2: Sum of Squares (AVX2 integer)

```
For each block of 32 int8 elements:
  1. Load 32 × int8 into __m256i
  2. Split into two 16 × int16 (cvtepi8_epi16 on low/high 128-bit halves)
  3. Square: mullo_epi16(v, v) → 16 × int16 squares
  4. Pair-sum to int32: madd_epi16(squares, ones_16) → 8 × int32
  5. Accumulate into __m256i int32 accumulator
After all blocks:
  6. Horizontal sum of 8 × int32 → single int32 sum_sq
```

Overflow check: max input ±80. Square max 6400. Pair-sum max 12800. 
Per-register (8 × int32): max 8 × 12800 × (2560/32 blocks) = 25.6M. Fits int32.

### Phase 3: Scalar rsqrt (FPU)

```
float mean = (float)sum_sq / n;
float scale = rsqrtf(mean + eps);  // or 1.0f / sqrtf(mean + eps)
```

One float division, one float sqrt. ~5 ns on Ryzen FPU. Called once.

To produce int8 output in range ±out_range, the effective scale is:
```
float effective_scale = scale * out_range;
// This maps: activation / rms(activations) * out_range → int8
```

Convert to fixed-point for the AVX2 scale multiply:
```
int16_t fixed_scale = (int16_t)roundf(effective_scale * 128);  // Q7 format
int shift = 7;  // right-shift after multiply to get back to int8 range
```

### Phase 4: Scale (AVX2 integer with fixed-point scalar)

```
__m256i vscale = _mm256_set1_epi16(fixed_scale);

For each block of 32 int8 elements:
  1. Load 32 × int8
  2. Split into two 16 × int16
  3. Multiply: mullo_epi16(v, vscale) → 16 × int16 products
  4. Arithmetic right-shift by 7: srai_epi16(prod, 7)
  5. Pack back to int8: packs_epi16(lo, hi) with saturation
  6. Store 32 × int8
```

The packs_epi16 saturates to [-128, +127], which covers all our output ranges.

### Alternative Phase 4: Scale via float (simpler, nearly as fast)

```
__m256 vscale = _mm256_set1_ps(scale);

For each block of 8 int8 elements:
  1. Widen int8 → int32 → float
  2. Multiply: mul_ps(v_float, vscale)
  3. Round: cvtps_epi32 (round to nearest)
  4. Pack int32 → int16 → int8 with saturation
```

This is 8 elements per cycle instead of 16, but avoids the fixed-point engineering.
For n=2560, this is 320 iterations × ~2 cycles = 640 cycles ≈ 210 ns. Still < 1 us.

## Expected Performance

| Phase | Operations | Estimated cycles | Estimated time (3 GHz) |
|-------|-----------|-----------------|----------------------|
| Square + accumulate | 80 × (load + widen + mul + madd + add) | ~320 | ~107 ns |
| Horizontal sum | 1 × hsum | ~10 | ~3 ns |
| rsqrt | 1 × sqrtf + div | ~15 | ~5 ns |
| Scale (fixed-point) | 80 × (load + widen + mul + shift + pack + store) | ~480 | ~160 ns |
| **Total** | | **~825** | **~275 ns** |

vs current MTFP21: 180,000 ns. That's **650x faster**.
vs float32 RMSNorm: 2,600 ns. That's **9x faster than float**.

The integer path is faster than float because:
- int8 is 4x denser than float32 (same data, fewer cache lines)
- mullo_epi16 processes 16 elements vs mulps's 8
- No float conversion overhead in the hot loop (fixed-point path)

## What MTFP21 Becomes

MTFP21 retains its role:
1. **Proof of concept:** Integer-only RMSNorm is possible at 25.4-bit precision (102/102 tests)
2. **Transport format:** Scale factors, gammas, epsilons stored in balanced ternary
3. **iGPU interface:** When softmax moves to iGPU, values pass in MTFP21
4. **Fallback:** Systems without SIMD or FPU can use the scalar MTFP21 path

The kernel redesign doesn't delete MTFP21. It puts it in its correct architectural layer.

## What to Build

1. `ternary_rmsnorm_avx2()` in a new file `bitnet/shirley_kernels.h`
2. Test: compare against float64 reference, same acceptance criteria as MTFP21 tests
3. Profile: measure actual ns/call, compare against float32 and MTFP21
4. Integration test: drop into the BitNet pipeline, measure PPL and generation quality

## iGPU: When and Where

Not for RMSNorm (rsqrt is 4% of pipeline, FPU handles it in 5 ns).

For softmax: 5.2M EXP operations per layer × 30 layers = 156M EXP per forward pass. At ~15 cycles per software EXP vs ~4 cycles per hardware V_EXP_F32, the iGPU saves ~11 cycles × 156M = 1.7B cycles ≈ 570 ms. THAT justifies the integration effort.

The iGPU roadmap:
1. Build the AVX2 integer kernel first (this synthesis)
2. Measure end-to-end inference with integer RMSNorm + float attention
3. Profile attention softmax — if it's the new bottleneck, integrate iGPU there
4. The iGPU integration for softmax is a separate LEMM cycle
