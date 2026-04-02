/*
 * shirley_mtfp16_matmul.h — Pure MTFP16 ternary matmul kernel
 *
 * int16 activations × 2-bit packed ternary weights → int32 accumulator
 * No float conversion anywhere. The matmul speaks integer.
 *
 * Uses sign_epi16 for ternary multiply (16 lanes per cycle).
 * Weight unpacking identical to ggml-bitnet-mad.cpp.
 */

#ifndef SHIRLEY_MTFP16_MATMUL_H
#define SHIRLEY_MTFP16_MATMUL_H

#include <stdint.h>
#include <immintrin.h>

/* Ternary dot product: int16 activations × 2-bit packed weights → int32
 *
 * n: inner dimension (must be multiple of 128 for the packed weight format)
 * act: int16 activations [n] (block-aligned MTFP16 mantissas)
 * weights: 2-bit packed ternary [n/4 bytes] (same format as ggml I2_S)
 *
 * Returns: int32 dot product (raw accumulator)
 *
 * The caller combines this with the block exponent and weight scale
 * to get the MTFP21 result. */
static inline int32_t shirley_ternary_dot_mtfp16(
    const int16_t * restrict act,
    const uint8_t * restrict weights,
    int n
) {
    const __m256i mask = _mm256_set1_epi8(0x03);
    const __m256i one8 = _mm256_set1_epi8(1);
    const __m256i one16 = _mm256_set1_epi16(1);

    __m256i acc = _mm256_setzero_si256();

    /* Process 128 elements per outer iteration (matches weight packing) */
    int n_groups = n / 128;

    for (int g = 0; g < n_groups; g++) {
        const uint8_t * pw = weights + g * 32;  /* 32 packed bytes = 128 weights */
        const int16_t * pa = act + g * 128;

        /* Unpack 2-bit weights: 32 bytes → 4×32 = 128 int8 ternary values */
        __m256i packed = _mm256_loadu_si256((const __m256i *)pw);
        __m256i w8_3 = _mm256_and_si256(packed, mask);
        __m256i w8_2 = _mm256_and_si256(_mm256_srli_epi16(packed, 2), mask);
        __m256i w8_1 = _mm256_and_si256(_mm256_srli_epi16(packed, 4), mask);
        __m256i w8_0 = _mm256_and_si256(_mm256_srli_epi16(packed, 6), mask);

        /* {0,1,2} → {-1,0,+1} */
        w8_0 = _mm256_sub_epi8(w8_0, one8);
        w8_1 = _mm256_sub_epi8(w8_1, one8);
        w8_2 = _mm256_sub_epi8(w8_2, one8);
        w8_3 = _mm256_sub_epi8(w8_3, one8);

        /* Process each group of 32 weights against 32 int16 activations.
         * Each int8→int16 extension gives us 16 values per register.
         * So each group of 32 weights needs 2 int16 loads. */

        /* Group 0: weights w8_0 (32 values), activations pa[0..31] */
        {
            __m128i w8_lo = _mm256_castsi256_si128(w8_0);
            __m128i w8_hi = _mm256_extracti128_si256(w8_0, 1);
            __m256i w16_lo = _mm256_cvtepi8_epi16(w8_lo);  /* 16 weights */
            __m256i w16_hi = _mm256_cvtepi8_epi16(w8_hi);  /* 16 weights */

            __m256i a16_lo = _mm256_loadu_si256((const __m256i *)(pa + 0));  /* 16 acts */
            __m256i a16_hi = _mm256_loadu_si256((const __m256i *)(pa + 16)); /* 16 acts */

            __m256i prod_lo = _mm256_sign_epi16(a16_lo, w16_lo);
            __m256i prod_hi = _mm256_sign_epi16(a16_hi, w16_hi);

            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(prod_lo, one16));
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(prod_hi, one16));
        }

        /* Group 1: weights w8_1, activations pa[32..63] */
        {
            __m128i w8_lo = _mm256_castsi256_si128(w8_1);
            __m128i w8_hi = _mm256_extracti128_si256(w8_1, 1);
            __m256i w16_lo = _mm256_cvtepi8_epi16(w8_lo);
            __m256i w16_hi = _mm256_cvtepi8_epi16(w8_hi);

            __m256i a16_lo = _mm256_loadu_si256((const __m256i *)(pa + 32));
            __m256i a16_hi = _mm256_loadu_si256((const __m256i *)(pa + 48));

            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_sign_epi16(a16_lo, w16_lo), one16));
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_sign_epi16(a16_hi, w16_hi), one16));
        }

        /* Group 2: weights w8_2, activations pa[64..95] */
        {
            __m128i w8_lo = _mm256_castsi256_si128(w8_2);
            __m128i w8_hi = _mm256_extracti128_si256(w8_2, 1);
            __m256i w16_lo = _mm256_cvtepi8_epi16(w8_lo);
            __m256i w16_hi = _mm256_cvtepi8_epi16(w8_hi);

            __m256i a16_lo = _mm256_loadu_si256((const __m256i *)(pa + 64));
            __m256i a16_hi = _mm256_loadu_si256((const __m256i *)(pa + 80));

            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_sign_epi16(a16_lo, w16_lo), one16));
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_sign_epi16(a16_hi, w16_hi), one16));
        }

        /* Group 3: weights w8_3, activations pa[96..127] */
        {
            __m128i w8_lo = _mm256_castsi256_si128(w8_3);
            __m128i w8_hi = _mm256_extracti128_si256(w8_3, 1);
            __m256i w16_lo = _mm256_cvtepi8_epi16(w8_lo);
            __m256i w16_hi = _mm256_cvtepi8_epi16(w8_hi);

            __m256i a16_lo = _mm256_loadu_si256((const __m256i *)(pa + 96));
            __m256i a16_hi = _mm256_loadu_si256((const __m256i *)(pa + 112));

            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_sign_epi16(a16_lo, w16_lo), one16));
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_sign_epi16(a16_hi, w16_hi), one16));
        }
    }

    /* Horizontal sum */
    __m128i lo = _mm256_castsi256_si128(acc);
    __m128i hi = _mm256_extracti128_si256(acc, 1);
    __m128i sum4 = _mm_add_epi32(lo, hi);
    sum4 = _mm_hadd_epi32(sum4, sum4);
    sum4 = _mm_hadd_epi32(sum4, sum4);

    return _mm_cvtsi128_si32(sum4);
}

/* Full MTFP16 gemv: one activation vector × weight matrix → output vector
 *
 * n_inner: activation dimension (e.g. 2560)
 * n_output: number of output rows (e.g. 6912 for FFN gate)
 * act_mant: block-aligned int16 mantissas [n_inner]
 * block_exp: shared exponent for the activation block
 * weight_data: packed 2-bit weight tensor data (ggml I2_S format)
 *              Layout: [n_output rows × n_inner/4 packed bytes] [float weight_scale]
 * weight_scale: the per-layer weight scale factor
 * dst: output MTFP21 values [n_output]
 */
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
    int row_bytes = n_inner / 4;  /* packed bytes per weight row */

    mtfp21_t ws = mtfp21_from_float(weight_scale);

    for (int row = 0; row < n_output; row++) {
        int32_t raw = shirley_ternary_dot_mtfp16(
            act_mant,
            weights + row * row_bytes,
            n_inner
        );

        /* Result: raw × 3^block_exp × weight_scale */
        mtfp21_t r;
        r.mantissa = raw;
        r.exponent = block_exp;

        /* Normalize mantissa */
        if (r.mantissa != 0) {
            while (llabs(r.mantissa) > MTFP21_MANT_MAX) {
                int32_t rem = r.mantissa % 3;
                r.mantissa /= 3;
                if (rem == 2) r.mantissa++;
                else if (rem == -2) r.mantissa--;
                r.exponent++;
            }
            while (llabs(r.mantissa) * 3 <= MTFP21_MANT_MAX &&
                   r.exponent > -MTFP21_EXP_MAX) {
                r.mantissa *= 3;
                r.exponent--;
            }
        }

        dst[row] = mtfp21_mul(r, ws);
    }
}

#endif /* SHIRLEY_MTFP16_MATMUL_H */
