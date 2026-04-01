/*
 * shirley_kernels.h — AVX2 integer kernels for ternary inference pipeline
 *
 * Layer 1 of the Shirley compute stack:
 *   Layer 1: AVX2 integer — per-element bulk (this file)
 *   Layer 2: FPU scalar — one rsqrt per RMSNorm
 *   Layer 3: MTFP21 — transport format
 *
 * Build: requires -mavx2 -mfma (or -march=native on AVX2 hardware)
 */

#ifndef SHIRLEY_KERNELS_H
#define SHIRLEY_KERNELS_H

#include <stdint.h>
#include <math.h>
#include <immintrin.h>

/* ================================================================
 *  ternary_rmsnorm_avx2 — Integer RMSNorm for ternary pipeline
 *
 *  Input:  int8 activations from matmul (range ±in_range)
 *  Output: int8 normalized activations (range ±out_range)
 *
 *  Per-element work: AVX2 integer (Layer 1)
 *  Scalar rsqrt: FPU float (Layer 2)
 *
 *  y[i] = clamp(round(x[i] / rms(x) * out_range / in_range), ±out_range)
 *  where rms(x) = sqrt(mean(x^2) + eps)
 * ================================================================ */

static void ternary_rmsnorm_avx2(
    int8_t       * restrict dst,
    const int8_t * restrict src,
    int           n,
    float         eps,
    int           out_range   /* output clamp, e.g. 80 for 5-trit */
) {
    /* ---- Phase 1+2: Sum of squares in AVX2 integer ----
     *
     * For each 32 int8 elements:
     *   Split into two 16×int16 halves
     *   Square: mullo_epi16(v, v)
     *   Pair-sum to int32: madd_epi16(sq, ones)
     *   Accumulate into int32 register
     *
     * Overflow: input ±80 max. Square max 6400. Pair-sum max 12800.
     * 80 blocks × 8 int32 lanes × 12800 = 81.9M max. Fits int32.
     */

    __m256i acc32 = _mm256_setzero_si256();
    const __m256i ones_16 = _mm256_set1_epi16(1);
    int i;

    for (i = 0; i + 32 <= n; i += 32) {
        __m256i v = _mm256_loadu_si256((const __m256i *)(src + i));

        /* Split into low and high 16×int16 */
        __m256i lo16 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(v));
        __m256i hi16 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(v, 1));

        /* Square */
        __m256i sq_lo = _mm256_mullo_epi16(lo16, lo16);
        __m256i sq_hi = _mm256_mullo_epi16(hi16, hi16);

        /* Pair-sum int16 → int32: (sq[0]+sq[1], sq[2]+sq[3], ...) */
        __m256i sum32_lo = _mm256_madd_epi16(sq_lo, ones_16);
        __m256i sum32_hi = _mm256_madd_epi16(sq_hi, ones_16);

        /* Accumulate into int32 */
        acc32 = _mm256_add_epi32(acc32, sum32_lo);
        acc32 = _mm256_add_epi32(acc32, sum32_hi);
    }

    /* Horizontal sum of 8×int32 → scalar */
    __m128i lo128 = _mm256_castsi256_si128(acc32);
    __m128i hi128 = _mm256_extracti128_si256(acc32, 1);
    __m128i sum128 = _mm_add_epi32(lo128, hi128);
    sum128 = _mm_hadd_epi32(sum128, sum128);
    sum128 = _mm_hadd_epi32(sum128, sum128);
    int32_t sum_sq = _mm_cvtsi128_si32(sum128);

    /* Handle tail */
    for (; i < n; i++) {
        sum_sq += (int32_t)src[i] * (int32_t)src[i];
    }

    /* ---- Phase 3: Scalar rsqrt (FPU — Layer 2) ----
     *
     * One float rsqrt. ~5 ns. Called once per RMSNorm.
     */

    float mean = (float)sum_sq / (float)n;
    float scale = 1.0f / sqrtf(mean + eps);

    /* Effective scale: maps input range to output range after normalization.
     * If input is in [-in_rms, +in_rms] (rms of input), output should be
     * in [-out_range, +out_range]. The scale factor is:
     *   output = round(input * scale * out_range_factor)
     * But we want the output to represent the normalized value quantized
     * to out_range, so: output = round(input * rsqrt(mean+eps))
     * clamped to ±out_range.
     *
     * For fixed-point: multiply int8 by (scale * 256), then shift right 8.
     * This gives Q8 fixed-point with ±1 precision.
     */

    /* Use float scaling path — simpler, nearly as fast for n=2560 */

    /* ---- Phase 4: Scale via float vectorized multiply ----
     *
     * _mm256_cvtepi8_epi32: widen 8 int8 → 8 int32
     * _mm256_cvtepi32_ps:   int32 → float
     * _mm256_mul_ps:        float × scale
     * _mm256_cvtps_epi32:   float → int32 (round to nearest)
     * _mm256_packs_epi32 + _mm256_packs_epi16: int32 → int16 → int8 with saturation
     */

    __m256 vscale = _mm256_set1_ps(scale);

    for (i = 0; i + 16 <= n; i += 16) {
        /* Process 16 elements in 2 groups of 8.
         * _mm256_cvtepi8_epi32 takes the low 8 bytes of a 128-bit input
         * and widens each to int32 in a 256-bit register. */
        __m128i src128 = _mm_loadu_si128((const __m128i *)(src + i));

        /* Group 0: elements [0..7] */
        __m256i vi32_0 = _mm256_cvtepi8_epi32(src128);
        __m256 vf_0 = _mm256_mul_ps(_mm256_cvtepi32_ps(vi32_0), vscale);
        __m256i vr32_0 = _mm256_cvtps_epi32(vf_0);  /* round to nearest int */

        /* Group 1: elements [8..15] */
        __m256i vi32_1 = _mm256_cvtepi8_epi32(_mm_srli_si128(src128, 8));
        __m256 vf_1 = _mm256_mul_ps(_mm256_cvtepi32_ps(vi32_1), vscale);
        __m256i vr32_1 = _mm256_cvtps_epi32(vf_1);

        /* Pack int32 → int16 → int8 with correct lane ordering.
         *
         * _mm256_packs_epi32(a, b) produces:
         *   [a_lo[0..3], b_lo[0..3], a_hi[0..3], b_hi[0..3]] as int16
         * where lo/hi refer to the 128-bit lanes.
         * To get elements in order, we permute after packing.
         */
        __m256i vi16 = _mm256_packs_epi32(vr32_0, vr32_1);
        /* Now vi16 has: [grp0_lo, grp1_lo, grp0_hi, grp1_hi] in 64-bit chunks.
         * We want: [grp0_lo, grp0_hi, grp1_lo, grp1_hi]. Permute: {0,2,1,3} */
        vi16 = _mm256_permute4x64_epi64(vi16, 0xD8);

        /* Clamp to ±out_range in int16 */
        vi16 = _mm256_max_epi16(vi16, _mm256_set1_epi16((int16_t)(-out_range)));
        vi16 = _mm256_min_epi16(vi16, _mm256_set1_epi16((int16_t)out_range));

        /* Pack int16 → int8. We only have 16 useful int16 values in the low
         * 16 positions. Use _mm256_packs_epi16 with zeros for the high input,
         * then extract the low 16 bytes. */
        __m256i vi8_32 = _mm256_packs_epi16(vi16, _mm256_setzero_si256());
        /* Result: [16 useful bytes, 0s, 16 useful bytes interleaved with 0s...]
         * Permute to get the 16 bytes contiguous. */
        vi8_32 = _mm256_permute4x64_epi64(vi8_32, 0xD8);

        /* Store low 16 bytes */
        _mm_storeu_si128((__m128i *)(dst + i), _mm256_castsi256_si128(vi8_32));
    }

    /* Handle tail */
    for (; i < n; i++) {
        float v = (float)src[i] * scale;
        int32_t r = (int32_t)roundf(v);
        if (r > out_range) r = out_range;
        if (r < -out_range) r = -out_range;
        dst[i] = (int8_t)r;
    }
}

#endif /* SHIRLEY_KERNELS_H */
