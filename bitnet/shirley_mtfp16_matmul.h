/*
 * shirley_mtfp16_matmul.h — Pure MTFP16 ternary matmul kernel
 *
 * int16 activations × 2-bit packed ternary weights → MTFP21 output
 * No float conversion. Batched normalization: all dot products first,
 * one global shift, per-element precision recovery.
 *
 * Uses sign_epi16 for ternary multiply (16 lanes per cycle).
 */

#ifndef SHIRLEY_MTFP16_MATMUL_H
#define SHIRLEY_MTFP16_MATMUL_H

#include <stdint.h>
#include <immintrin.h>

/* Single ternary dot product: int16 activations × 2-bit packed weights → int32 */
static inline int32_t shirley_ternary_dot_mtfp16(
    const int16_t * restrict act,
    const uint8_t * restrict weights,
    int n
) {
    const __m256i mask = _mm256_set1_epi8(0x03);
    const __m256i one8 = _mm256_set1_epi8(1);
    const __m256i one16 = _mm256_set1_epi16(1);

    __m256i acc = _mm256_setzero_si256();
    int n_groups = n / 128;

    for (int g = 0; g < n_groups; g++) {
        const uint8_t * pw = weights + g * 32;
        const int16_t * pa = act + g * 128;

        __m256i packed = _mm256_loadu_si256((const __m256i *)pw);
        __m256i w8_3 = _mm256_and_si256(packed, mask);
        __m256i w8_2 = _mm256_and_si256(_mm256_srli_epi16(packed, 2), mask);
        __m256i w8_1 = _mm256_and_si256(_mm256_srli_epi16(packed, 4), mask);
        __m256i w8_0 = _mm256_and_si256(_mm256_srli_epi16(packed, 6), mask);

        w8_0 = _mm256_sub_epi8(w8_0, one8);
        w8_1 = _mm256_sub_epi8(w8_1, one8);
        w8_2 = _mm256_sub_epi8(w8_2, one8);
        w8_3 = _mm256_sub_epi8(w8_3, one8);

        /* 4 groups of 32 weights, each processed as 2×16 int16 */
        #define PROCESS_GROUP(w8, offset) do { \
            __m128i w8_lo = _mm256_castsi256_si128(w8); \
            __m128i w8_hi = _mm256_extracti128_si256(w8, 1); \
            __m256i w16_lo = _mm256_cvtepi8_epi16(w8_lo); \
            __m256i w16_hi = _mm256_cvtepi8_epi16(w8_hi); \
            __m256i a16_lo = _mm256_loadu_si256((const __m256i *)(pa + offset)); \
            __m256i a16_hi = _mm256_loadu_si256((const __m256i *)(pa + offset + 16)); \
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_sign_epi16(a16_lo, w16_lo), one16)); \
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_sign_epi16(a16_hi, w16_hi), one16)); \
        } while(0)

        PROCESS_GROUP(w8_0, 0);
        PROCESS_GROUP(w8_1, 32);
        PROCESS_GROUP(w8_2, 64);
        PROCESS_GROUP(w8_3, 96);

        #undef PROCESS_GROUP
    }

    __m128i lo = _mm256_castsi256_si128(acc);
    __m128i hi = _mm256_extracti128_si256(acc, 1);
    __m128i sum4 = _mm_add_epi32(lo, hi);
    sum4 = _mm_hadd_epi32(sum4, sum4);
    sum4 = _mm_hadd_epi32(sum4, sum4);

    return _mm_cvtsi128_si32(sum4);
}

/* ================================================================
 *  Batched ternary gemv: all dot products first, then batch normalize
 *
 *  Phase 1: compute all raw int32 dot products (pure SIMD)
 *  Phase 2: find global max absolute value (one SIMD pass)
 *  Phase 3: compute global trit-shift (one scalar op)
 *  Phase 4: apply shift + precision recovery + weight scale (one pass)
 *
 *  Replaces 307,200 individual normalization sequences per token
 *  with 3 bulk passes over the output array.
 * ================================================================ */

static inline void shirley_gemv_mtfp16(
    mtfp21_t * restrict dst,
    const int16_t * restrict act_mant,
    int8_t block_exp,
    const void * restrict weight_data,
    int n_inner,
    int n_output,
    float weight_scale
) {
    const uint8_t * weights = (const uint8_t *)weight_data;
    int row_bytes = n_inner / 4;

    /* Phase 1: all dot products → raw int32 array (pure SIMD) */
    int32_t raw[n_output]; /* VLA */
    for (int row = 0; row < n_output; row++) {
        raw[row] = shirley_ternary_dot_mtfp16(
            act_mant, weights + row * row_bytes, n_inner);
    }

    /* Phase 2: find global max absolute value */
    int32_t max_abs = 0;
    for (int i = 0; i < n_output; i++) {
        int32_t a = raw[i] > 0 ? raw[i] : -raw[i];
        if (a > max_abs) max_abs = a;
    }

    if (max_abs == 0) {
        for (int i = 0; i < n_output; i++) dst[i] = (mtfp21_t){0, 0};
        return;
    }

    /* Phase 3: compute global trit-shift */
    int shift_down = 0;
    {
        int64_t test = (int64_t)max_abs;
        while (test > MTFP21_MANT_MAX) {
            test /= 3;
            shift_down++;
        }
    }

    /* Phase 4: apply shift + precision recovery + weight scale */
    mtfp21_t ws = mtfp21_from_float(weight_scale);
    int64_t divisor = (shift_down > 0 && shift_down < 32) ? POW3[shift_down] : 1;
    int64_t half = divisor / 2;

    for (int i = 0; i < n_output; i++) {
        if (raw[i] == 0) { dst[i] = (mtfp21_t){0, 0}; continue; }

        /* Global shift-down (uniform across all rows) */
        int64_t m = (int64_t)raw[i];
        int exp = (int)block_exp;

        if (shift_down > 0) {
            if (m >= 0) m = (m + half) / divisor;
            else        m = -((-m + half) / divisor);
            exp += shift_down;
        }

        /* Per-element precision recovery: shift up to maximize mantissa */
        while (m != 0 && llabs(m) * 3 <= MTFP21_MANT_MAX && exp > -MTFP21_EXP_MAX) {
            m *= 3;
            exp--;
        }

        /* Apply weight scale */
        mtfp21_t r;
        r.mantissa = (int32_t)m;
        r.exponent = (int8_t)exp;
        dst[i] = mtfp21_mul(r, ws);
    }
}

#endif /* SHIRLEY_MTFP16_MATMUL_H */
