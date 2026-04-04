/*
 * shirley_mtfp16_ops.h — SIMD operations on MTFP16/21 values
 *
 * Adaptive-width MTFP: int16 mantissa for 16-lane SIMD,
 * int32 for overflow-prone ops, all with base-3 exponents.
 *
 * Build: requires -mavx2 (or -march=native on AVX2 hardware)
 */

#ifndef SHIRLEY_MTFP16_OPS_H
#define SHIRLEY_MTFP16_OPS_H

#include <stdint.h>
#include <immintrin.h>
#include "shirley_mtfp21.h"

#define MTFP16_MANT_MAX_OPS 29524

/* ================================================================
 *  Block-align MTFP21 array to shared exponent → int16 mantissas
 * ================================================================ */

static inline int8_t mtfp16_block_align_vec(
    int16_t * dst, const mtfp21_t * src, int n
) {
    /* Find max exponent */
    int8_t max_exp = -128;
    for (int i = 0; i < n; i++) {
        if (src[i].mantissa != 0 && src[i].exponent > max_exp)
            max_exp = src[i].exponent;
    }
    if (max_exp == -128) max_exp = 0;

    /* Truncate + align */
    for (int i = 0; i < n; i++) {
        if (src[i].mantissa == 0) { dst[i] = 0; continue; }

        int32_t m = src[i].mantissa;
        int shift = max_exp - src[i].exponent;

        /* First truncate to MTFP16 range */
        while (m > MTFP16_MANT_MAX_OPS || m < -MTFP16_MANT_MAX_OPS) {
            int32_t rem = m % 3; m /= 3;
            if (rem == 2) m++; else if (rem == -2) m--;
            shift--;
        }

        /* Then align to shared exponent */
        for (int s = 0; s < shift && s < 20; s++) {
            int32_t rem = m % 3; m /= 3;
            if (rem == 2) m++; else if (rem == -2) m--;
        }
        dst[i] = (int16_t)m;
    }
    return max_exp;
}

/* ================================================================
 *  MTFP16 SIMD dot product: int16 × int16 → int32 accumulator
 *
 *  For Q@K^T and attn@V where BOTH operands are continuous.
 *  Uses madd_epi16: a[2i]*b[2i] + a[2i+1]*b[2i+1] → int32
 *  16 multiplies + 8 adds per cycle.
 *
 *  Returns int32 raw result. Caller combines with exponents:
 *    real = raw × 3^(exp_a + exp_b)
 * ================================================================ */

static inline int32_t mtfp16_dot_simd(
    const int16_t * restrict a,
    const int16_t * restrict b,
    int n
) {
    __m256i acc = _mm256_setzero_si256();
    int i;

    for (i = 0; i + 16 <= n; i += 16) {
        __m256i va = _mm256_loadu_si256((const __m256i *)(a + i));
        __m256i vb = _mm256_loadu_si256((const __m256i *)(b + i));
        /* madd: a[0]*b[0]+a[1]*b[1], a[2]*b[2]+a[3]*b[3], ... → 8 int32 */
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(va, vb));
    }

    /* Horizontal sum */
    __m128i lo = _mm256_castsi256_si128(acc);
    __m128i hi = _mm256_extracti128_si256(acc, 1);
    __m128i sum4 = _mm_add_epi32(lo, hi);
    sum4 = _mm_hadd_epi32(sum4, sum4);
    sum4 = _mm_hadd_epi32(sum4, sum4);
    int32_t total = _mm_cvtsi128_si32(sum4);

    /* Scalar tail */
    for (; i < n; i++) {
        total += (int32_t)a[i] * (int32_t)b[i];
    }

    return total;
}

/* ================================================================
 *  MTFP16 SIMD ReLU: max(x, 0) — 16 lanes
 * ================================================================ */

static inline void mtfp16_relu_simd(
    int16_t * dst_mant, int8_t * dst_exp,
    const int16_t * src_mant, const int8_t * src_exp,
    int n
) {
    const __m256i zero = _mm256_setzero_si256();
    int i;

    for (i = 0; i + 16 <= n; i += 16) {
        __m256i v = _mm256_loadu_si256((const __m256i *)(src_mant + i));
        _mm256_storeu_si256((__m256i *)(dst_mant + i), _mm256_max_epi16(v, zero));
        /* Zero exponents where mantissa was zeroed */
        for (int j = i; j < i + 16; j++) {
            dst_exp[j] = (dst_mant[j] != 0) ? src_exp[j] : 0;
        }
    }
    for (; i < n; i++) {
        if (src_mant[i] > 0) {
            dst_mant[i] = src_mant[i];
            dst_exp[i] = src_exp[i];
        } else {
            dst_mant[i] = 0;
            dst_exp[i] = 0;
        }
    }
}

/* ================================================================
 *  MTFP16 Square: x² — widen to int32, 8 lanes
 *
 *  mantissa² fits int32 for MTFP16 mantissa ≤ 29524 (29524² = 871M < 2^31)
 *  Actually 29524² = 871,665,576 which DOES fit int32 (max 2,147,483,647).
 *  Exponent doubles.
 *  Output: int32 mantissa + int8 exponent (MTFP21-compatible)
 * ================================================================ */

static inline void mtfp16_square_simd(
    int32_t * dst_mant, int8_t * dst_exp,
    const int16_t * src_mant, const int8_t * src_exp,
    int n
) {
    int i;
    for (i = 0; i + 8 <= n; i += 8) {
        __m128i v16 = _mm_loadu_si128((const __m128i *)(src_mant + i));  /* 8 int16 */
        __m256i v32 = _mm256_cvtepi16_epi32(v16);                        /* widen to int32 */
        __m256i sq = _mm256_mullo_epi32(v32, v32);                        /* square */
        _mm256_storeu_si256((__m256i *)(dst_mant + i), sq);

        for (int j = i; j < i + 8; j++) {
            dst_exp[j] = (int8_t)((int)src_exp[j] * 2);  /* exponent doubles */
        }
    }
    for (; i < n; i++) {
        dst_mant[i] = (int32_t)src_mant[i] * (int32_t)src_mant[i];
        dst_exp[i] = (int8_t)((int)src_exp[i] * 2);
    }
}

/* ================================================================
 *  MTFP element-wise multiply: a × b → result
 *
 *  a is int32 mantissa (from square), b is int16 mantissa.
 *  Product may overflow int32. Use int64 per element (scalar).
 *  Output as MTFP21.
 * ================================================================ */

static inline void mtfp_elem_mul_32x16(
    mtfp21_t * dst,
    const int32_t * a_mant, const int8_t * a_exp,
    const int16_t * b_mant, const int8_t * b_exp,
    int n
) {
    for (int i = 0; i < n; i++) {
        if (a_mant[i] == 0 || b_mant[i] == 0) {
            dst[i] = (mtfp21_t){0, 0};
            continue;
        }
        int64_t prod = (int64_t)a_mant[i] * (int64_t)b_mant[i];
        int exp = (int)a_exp[i] + (int)b_exp[i];

        /* Normalize to MTFP21 range */
        while (llabs(prod) > MTFP21_MANT_MAX) {
            int64_t rem = prod % 3;
            prod /= 3;
            if (rem == 2) prod++; else if (rem == -2) prod--;
            exp++;
        }
        while (prod != 0 && llabs(prod) * 3 <= MTFP21_MANT_MAX && exp > -MTFP21_EXP_MAX) {
            prod *= 3;
            exp--;
        }

        dst[i].mantissa = (int32_t)prod;
        dst[i].exponent = (int8_t)exp;
    }
}

/* ================================================================
 *  MTFP16 SIMD sum-of-squares for RMSNorm
 *
 *  madd_epi16(x, x) = x[0]²+x[1]², x[2]²+x[3]², ... → 8 int32
 *  Accumulates into int32. For n=2560 with max mantissa 29524:
 *  worst case sum = 2560 × 29524² ≈ 2.2e12 — overflows int32.
 *  Use int64 accumulator or process in blocks.
 *
 *  Approach: accumulate int32 pairs, widen to int64 periodically.
 * ================================================================ */

static inline int64_t mtfp16_sum_of_squares(
    const int16_t * src, int n
) {
    int64_t total = 0;
    __m256i acc = _mm256_setzero_si256();
    int i;
    int block = 0;

    for (i = 0; i + 16 <= n; i += 16) {
        __m256i v = _mm256_loadu_si256((const __m256i *)(src + i));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(v, v));
        block += 16;

        /* Flush to int64 every 256 elements to avoid int32 overflow */
        if (block >= 256) {
            int32_t tmp[8];
            _mm256_storeu_si256((__m256i *)tmp, acc);
            for (int j = 0; j < 8; j++) total += (int64_t)tmp[j];
            acc = _mm256_setzero_si256();
            block = 0;
        }
    }

    /* Flush remaining accumulator */
    {
        int32_t tmp[8];
        _mm256_storeu_si256((__m256i *)tmp, acc);
        for (int j = 0; j < 8; j++) total += (int64_t)tmp[j];
    }

    /* Scalar tail */
    for (; i < n; i++) {
        total += (int64_t)src[i] * (int64_t)src[i];
    }

    return total;
}

#endif /* SHIRLEY_MTFP16_OPS_H */
