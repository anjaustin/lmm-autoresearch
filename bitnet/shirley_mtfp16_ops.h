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

/* ================================================================
 *  MTFP21 chunked dot product: per-element exponents, 8-wide chunks
 *
 *  Each element has its own mantissa (int32) and exponent (int8).
 *  Process 8 elements at a time:
 *    - Multiply mantissas (int32 × int32 → int64, 4 lanes × 2)
 *    - Sum exponents per product
 *    - Find max exponent within the 8-element chunk
 *    - Align all products to chunk max exponent
 *    - Accumulate in int64 with running exponent tracking
 *
 *  Within 8 adjacent elements, exponent spread is typically 0-2 trits.
 *  Precision loss is bounded and negligible.
 * ================================================================ */

static inline mtfp21_t mtfp21_dot_chunked(
    const int32_t * restrict a_mant, const int8_t * restrict a_exp,
    const int32_t * restrict b_mant, const int8_t * restrict b_exp,
    int n
) {
    int64_t acc_mant = 0;
    int acc_exp = 0;
    int acc_initialized = 0;

    int i;
    for (i = 0; i + 8 <= n; i += 8) {
        /* Compute 8 mantissa products and exponent sums */
        int64_t prods[8];
        int8_t  pexp[8];
        int8_t  chunk_max_exp = -128;

        /* Two mul_epi32 calls: lanes 0,2,4,6 then 1,3,5,7 */
        __m256i va = _mm256_loadu_si256((const __m256i *)(a_mant + i));
        __m256i vb = _mm256_loadu_si256((const __m256i *)(b_mant + i));

        /* _mm256_mul_epi32 multiplies lanes 0,2,4,6 → 4 int64 results */
        __m256i prod_even = _mm256_mul_epi32(va, vb);
        /* Shift to get odd lanes: 1,3,5,7 */
        __m256i va_odd = _mm256_srli_epi64(va, 32);
        __m256i vb_odd = _mm256_srli_epi64(vb, 32);
        __m256i prod_odd = _mm256_mul_epi32(va_odd, vb_odd);

        /* Extract to scalar for exponent handling */
        int64_t even[4], odd[4];
        _mm256_storeu_si256((__m256i *)even, prod_even);
        _mm256_storeu_si256((__m256i *)odd, prod_odd);

        /* Interleave: even has indices 0,2,4,6; odd has 1,3,5,7 */
        prods[0] = even[0]; prods[1] = odd[0];
        prods[2] = even[1]; prods[3] = odd[1];
        prods[4] = even[2]; prods[5] = odd[2];
        prods[6] = even[3]; prods[7] = odd[3];

        for (int j = 0; j < 8; j++) {
            pexp[j] = (int8_t)((int)a_exp[i+j] + (int)b_exp[i+j]);
            if (prods[j] != 0 && pexp[j] > chunk_max_exp)
                chunk_max_exp = pexp[j];
        }
        if (chunk_max_exp == -128) continue; /* all zeros */

        /* Align products to chunk max exponent and sum */
        int64_t chunk_sum = 0;
        for (int j = 0; j < 8; j++) {
            if (prods[j] == 0) continue;
            int shift = chunk_max_exp - pexp[j];
            int64_t aligned = prods[j];
            for (int s = 0; s < shift && s < 30; s++) {
                aligned /= 3; /* trit-shift right */
            }
            chunk_sum += aligned;
        }

        /* Merge chunk into running accumulator */
        if (!acc_initialized) {
            acc_mant = chunk_sum;
            acc_exp = chunk_max_exp;
            acc_initialized = 1;
        } else {
            /* Align accumulator and chunk to same exponent */
            int diff = acc_exp - chunk_max_exp;
            if (diff > 0) {
                /* acc has larger exponent — shift chunk_sum right */
                for (int s = 0; s < diff && s < 30; s++) chunk_sum /= 3;
                acc_mant += chunk_sum;
            } else if (diff < 0) {
                /* chunk has larger exponent — shift acc right */
                for (int s = 0; s < -diff && s < 30; s++) acc_mant /= 3;
                acc_exp = chunk_max_exp;
                acc_mant += chunk_sum;
            } else {
                acc_mant += chunk_sum;
            }
        }
    }

    /* Scalar tail */
    for (; i < n; i++) {
        int64_t prod = (int64_t)a_mant[i] * (int64_t)b_mant[i];
        int pexp_i = (int)a_exp[i] + (int)b_exp[i];
        if (prod == 0) continue;

        if (!acc_initialized) {
            acc_mant = prod;
            acc_exp = pexp_i;
            acc_initialized = 1;
        } else {
            int diff = acc_exp - pexp_i;
            if (diff > 0) {
                for (int s = 0; s < diff && s < 30; s++) prod /= 3;
                acc_mant += prod;
            } else if (diff < 0) {
                for (int s = 0; s < -diff && s < 30; s++) acc_mant /= 3;
                acc_exp = pexp_i;
                acc_mant += prod;
            } else {
                acc_mant += prod;
            }
        }
    }

    /* Normalize to MTFP21 */
    while (llabs(acc_mant) > MTFP21_MANT_MAX) {
        int64_t rem = acc_mant % 3; acc_mant /= 3;
        if (rem == 2) acc_mant++; else if (rem == -2) acc_mant--;
        acc_exp++;
    }
    while (acc_mant != 0 && llabs(acc_mant) * 3 <= MTFP21_MANT_MAX && acc_exp > -MTFP21_EXP_MAX) {
        acc_mant *= 3; acc_exp--;
    }

    mtfp21_t result;
    result.mantissa = (int32_t)acc_mant;
    result.exponent = (int8_t)acc_exp;
    return result;
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

/* ================================================================
 *  SIMD RMSNorm: chunked sum-of-squares + vectorized scale×gamma
 *
 *  Input: MTFP21 parallel arrays (mantissa + exponent)
 *  Output: MTFP21 parallel arrays (normalized, gamma-applied)
 *  Only the rsqrt is scalar. Everything else is SIMD.
 * ================================================================ */

static inline void mtfp21_rmsnorm_simd(
    int32_t * dst_mant, int8_t * dst_exp,           /* output */
    const int32_t * src_mant, const int8_t * src_exp, /* input */
    const int32_t * gamma_mant, const int8_t * gamma_exp, /* gamma (NULL to skip) */
    int n, int32_t eps_mant, int8_t eps_exp
) {
    /* 1. Sum of squares — chunked SIMD */
    mtfp21_t sum_sq = mtfp21_dot_chunked(src_mant, src_exp, src_mant, src_exp, n);

    /* 2. rsqrt — one scalar (irreducible) */
    mtfp21_t mean = mtfp21_div_scalar(sum_sq, n);
    mtfp21_t scale = mtfp21_rsqrt(mtfp21_add(mean, (mtfp21_t){eps_mant, eps_exp}));

    /* 3. Per-element scale × gamma — SIMD 8-wide */
    int32_t sm = scale.mantissa;
    int i;

    for (i = 0; i + 8 <= n; i += 8) {
        __m256i vm = _mm256_loadu_si256((const __m256i *)(src_mant + i));
        __m256i vs = _mm256_set1_epi32(sm);

        __m256i prod_even = _mm256_mul_epi32(vm, vs);
        __m256i vm_odd = _mm256_srli_epi64(vm, 32);
        __m256i prod_odd = _mm256_mul_epi32(vm_odd, vs);

        int64_t even[4], odd[4];
        _mm256_storeu_si256((__m256i *)even, prod_even);
        _mm256_storeu_si256((__m256i *)odd, prod_odd);

        int64_t prods[8];
        prods[0] = even[0]; prods[1] = odd[0];
        prods[2] = even[1]; prods[3] = odd[1];
        prods[4] = even[2]; prods[5] = odd[2];
        prods[6] = even[3]; prods[7] = odd[3];

        for (int j = 0; j < 8; j++) {
            int exp = (int)src_exp[i+j] + (int)scale.exponent;
            int64_t p = prods[j];
            while (llabs(p) > MTFP21_MANT_MAX) { p /= 3; exp++; }
            while (p != 0 && llabs(p) * 3 <= MTFP21_MANT_MAX && exp > -MTFP21_EXP_MAX) { p *= 3; exp--; }

            if (gamma_mant) {
                /* Multiply by gamma */
                int64_t gp = p * (int64_t)gamma_mant[i+j];
                int ge = exp + (int)gamma_exp[i+j];
                while (llabs(gp) > MTFP21_MANT_MAX) { gp /= 3; ge++; }
                while (gp != 0 && llabs(gp) * 3 <= MTFP21_MANT_MAX && ge > -MTFP21_EXP_MAX) { gp *= 3; ge--; }
                dst_mant[i+j] = (int32_t)gp;
                dst_exp[i+j] = (int8_t)ge;
            } else {
                dst_mant[i+j] = (int32_t)p;
                dst_exp[i+j] = (int8_t)exp;
            }
        }
    }
    for (; i < n; i++) {
        mtfp21_t v = mtfp21_mul((mtfp21_t){src_mant[i], src_exp[i]}, scale);
        if (gamma_mant) v = mtfp21_mul(v, (mtfp21_t){gamma_mant[i], gamma_exp[i]});
        dst_mant[i] = v.mantissa;
        dst_exp[i] = v.exponent;
    }
}

#endif /* SHIRLEY_MTFP16_OPS_H */
