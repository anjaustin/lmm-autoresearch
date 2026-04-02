/*
 * test_mtfp16_matmul.c — Validate pure MTFP matmul math
 *
 * Tests: MTFP16 mantissa × ternary weight → int32 accumulator → MTFP result
 * No float conversion in the matmul path.
 *
 * Build: gcc -O2 -mavx2 -o test_mtfp16_matmul test_mtfp16_matmul.c -lm
 * Run:   ./test_mtfp16_matmul
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <immintrin.h>
#include "shirley_mtfp21.h"

static int tests_passed = 0;
static int tests_failed = 0;

/* ================================================================
 *  MTFP16: reduced-width MTFP for SIMD bulk compute
 * ================================================================ */

typedef struct {
    int16_t mantissa;   /* 10-trit range: ±29524 */
    int8_t  exponent;
} mtfp16_t;

#define MTFP16_MANT_MAX 29524  /* (3^10 - 1) / 2 */

/* MTFP21 → MTFP16: truncate mantissa with rounding */
static mtfp16_t mtfp21_to_mtfp16(mtfp21_t a) {
    mtfp16_t r;
    if (a.mantissa == 0) {
        r.mantissa = 0;
        r.exponent = 0;
        return r;
    }

    int32_t mant = a.mantissa;
    int exp = a.exponent;

    /* Trit-shift right until mantissa fits in int16 range */
    while (mant > MTFP16_MANT_MAX || mant < -MTFP16_MANT_MAX) {
        int32_t rem = mant % 3;
        mant = mant / 3;
        if (rem == 2) mant++;
        else if (rem == -2) mant--;
        exp++;
    }

    r.mantissa = (int16_t)mant;
    r.exponent = (int8_t)exp;
    return r;
}

/* MTFP16 → MTFP21: widen mantissa (lossless) */
static mtfp21_t mtfp16_to_mtfp21(mtfp16_t a) {
    mtfp21_t r;
    r.mantissa = (int32_t)a.mantissa;
    r.exponent = a.exponent;

    /* Normalize: maximize mantissa precision */
    while (r.mantissa != 0 &&
           (int64_t)r.mantissa * 3 <= MTFP21_MANT_MAX &&
           (int64_t)r.mantissa * 3 >= -MTFP21_MANT_MAX &&
           r.exponent > -MTFP21_EXP_MAX) {
        r.mantissa *= 3;
        r.exponent--;
    }

    return r;
}

/* MTFP16 → float (for verification) */
static float mtfp16_to_float(mtfp16_t a) {
    return mtfp21_to_float(mtfp16_to_mtfp21(a));
}

/* float → MTFP16 */
static mtfp16_t mtfp16_from_float(float f) {
    return mtfp21_to_mtfp16(mtfp21_from_float(f));
}

/* ================================================================
 *  Block exponent: extract shared exponent from MTFP16 vector
 *
 *  After RMSNorm, values have similar magnitude. Align to
 *  a shared exponent so mantissas can be used directly in SIMD.
 * ================================================================ */

typedef struct {
    int16_t * mantissas;   /* [n] aligned mantissas */
    int8_t    block_exp;   /* shared exponent */
} mtfp16_block_t;

/* Align MTFP16 vector to shared exponent.
 * Returns the block exponent. Mantissas adjusted accordingly. */
static int8_t mtfp16_align_block(
    int16_t * dst_mant,
    const mtfp16_t * src,
    int n
) {
    /* Find maximum exponent — all values align to this */
    int8_t max_exp = -128;
    for (int i = 0; i < n; i++) {
        if (src[i].mantissa != 0 && src[i].exponent > max_exp)
            max_exp = src[i].exponent;
    }
    if (max_exp == -128) max_exp = 0;

    /* Align each mantissa: shift right by (max_exp - elem_exp) trits */
    for (int i = 0; i < n; i++) {
        if (src[i].mantissa == 0) {
            dst_mant[i] = 0;
            continue;
        }
        int shift = max_exp - src[i].exponent;
        if (shift == 0) {
            dst_mant[i] = src[i].mantissa;
        } else if (shift > 0 && shift < 20) {
            /* Trit right-shift: divide by 3^shift with rounding */
            int32_t m = (int32_t)src[i].mantissa;
            for (int s = 0; s < shift; s++) {
                int32_t rem = m % 3;
                m = m / 3;
                if (rem == 2) m++;
                else if (rem == -2) m--;
            }
            dst_mant[i] = (int16_t)m;
        } else {
            dst_mant[i] = 0; /* shifted to zero */
        }
    }

    return max_exp;
}

/* ================================================================
 *  Pure MTFP ternary matmul — no float conversion
 *
 *  activation: MTFP16 vector [n]
 *  weights: ternary {-1, 0, +1} as int8 [n]
 *  result: MTFP21 scalar (the dot product)
 *
 *  Path A: per-element exponents (precise, scalar)
 *  Path B: block exponent (SIMD-friendly)
 * ================================================================ */

/* Path A: per-element exponents, scalar */
static mtfp21_t mtfp_ternary_dot_scalar(
    const mtfp16_t * act,
    const int8_t * weights,  /* {-1, 0, +1} */
    int n
) {
    mtfp21_t sum = {0, 0};
    for (int i = 0; i < n; i++) {
        if (weights[i] == 0) continue;
        mtfp21_t val;
        val.mantissa = (int32_t)act[i].mantissa * (int32_t)weights[i];
        val.exponent = act[i].exponent;
        sum = mtfp21_add(sum, val);
    }
    return sum;
}

/* Path B: block exponent, SIMD-ready (scalar reference impl) */
static mtfp21_t mtfp_ternary_dot_block(
    const int16_t * act_mant,  /* block-aligned mantissas */
    int8_t block_exp,
    const int8_t * weights,    /* {-1, 0, +1} */
    int n
) {
    /* sign_epi16 equivalent: ternary multiply on int16 mantissas */
    int32_t acc = 0;
    for (int i = 0; i < n; i++) {
        int16_t product;
        if (weights[i] > 0)       product = act_mant[i];
        else if (weights[i] == 0) product = 0;
        else                      product = -act_mant[i];
        acc += (int32_t)product;
    }

    /* Result is int32 mantissa with the block exponent */
    mtfp21_t result;
    result.mantissa = acc;
    result.exponent = block_exp;

    /* Normalize */
    while (result.mantissa != 0 &&
           llabs(result.mantissa) * 3 <= MTFP21_MANT_MAX &&
           result.exponent > -MTFP21_EXP_MAX) {
        result.mantissa *= 3;
        result.exponent--;
    }
    while (llabs(result.mantissa) > MTFP21_MANT_MAX &&
           result.exponent < MTFP21_EXP_MAX) {
        int32_t rem = result.mantissa % 3;
        result.mantissa /= 3;
        if (rem == 2) result.mantissa++;
        else if (rem == -2) result.mantissa--;
        result.exponent++;
    }

    return result;
}

/* Path B with AVX2: actual SIMD implementation */
static mtfp21_t mtfp_ternary_dot_avx2(
    const int16_t * act_mant,  /* block-aligned mantissas, 16-aligned */
    int8_t block_exp,
    const int8_t * weights,    /* {-1, 0, +1} ternary */
    int n
) {
    __m256i acc = _mm256_setzero_si256();
    const __m256i ones = _mm256_set1_epi16(1);
    int i;

    for (i = 0; i + 16 <= n; i += 16) {
        /* Load 16 int16 mantissas */
        __m256i a = _mm256_loadu_si256((const __m256i *)(act_mant + i));

        /* Load 16 int8 weights, sign-extend to int16 */
        __m128i w8 = _mm_loadu_si128((const __m128i *)(weights + i));
        __m256i w16 = _mm256_cvtepi8_epi16(w8);

        /* Ternary multiply: sign_epi16(activation, weight) */
        __m256i prod = _mm256_sign_epi16(a, w16);

        /* Accumulate: pair-wise horizontal add int16 → int32 */
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(prod, ones));
    }

    /* Horizontal sum of int32 accumulator */
    __m128i lo = _mm256_castsi256_si128(acc);
    __m128i hi = _mm256_extracti128_si256(acc, 1);
    __m128i sum4 = _mm_add_epi32(lo, hi);
    sum4 = _mm_hadd_epi32(sum4, sum4);
    sum4 = _mm_hadd_epi32(sum4, sum4);
    int32_t total = _mm_cvtsi128_si32(sum4);

    /* Scalar tail */
    for (; i < n; i++) {
        int16_t product;
        if (weights[i] > 0)       product = act_mant[i];
        else if (weights[i] == 0) product = 0;
        else                      product = -act_mant[i];
        total += (int32_t)product;
    }

    /* Wrap as MTFP21 */
    mtfp21_t result;
    result.mantissa = total;
    result.exponent = block_exp;

    /* Normalize */
    while (result.mantissa != 0 &&
           llabs(result.mantissa) * 3 <= MTFP21_MANT_MAX &&
           result.exponent > -MTFP21_EXP_MAX) {
        result.mantissa *= 3;
        result.exponent--;
    }
    while (llabs(result.mantissa) > MTFP21_MANT_MAX) {
        int32_t rem = result.mantissa % 3;
        result.mantissa /= 3;
        if (rem == 2) result.mantissa++;
        else if (rem == -2) result.mantissa--;
        result.exponent++;
    }

    return result;
}

/* ================================================================
 *  Float reference: standard dot product for comparison
 * ================================================================ */

static float float_ternary_dot(
    const float * act,
    const int8_t * weights,
    int n
) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += act[i] * (float)weights[i];
    }
    return sum;
}

/* ================================================================
 *  Tests
 * ================================================================ */

static void test_mtfp16_conversion(void) {
    printf("--- test_mtfp16_conversion ---\n");

    float test_vals[] = {1.0f, -1.0f, 0.5f, 100.0f, 0.001f, -42.7f, 0.0f};
    int n = sizeof(test_vals) / sizeof(test_vals[0]);
    int pass = 0;

    for (int i = 0; i < n; i++) {
        mtfp21_t m21 = mtfp21_from_float(test_vals[i]);
        mtfp16_t m16 = mtfp21_to_mtfp16(m21);
        mtfp21_t m21_back = mtfp16_to_mtfp21(m16);
        float recovered = mtfp21_to_float(m21_back);

        float rel_err = (test_vals[i] == 0.0f) ? fabsf(recovered)
            : fabsf((recovered - test_vals[i]) / test_vals[i]);

        printf("  %.4f → MTFP21{%d,%d} → MTFP16{%d,%d} → %.4f (err=%.2e)\n",
               test_vals[i], m21.mantissa, m21.exponent,
               m16.mantissa, m16.exponent, recovered, rel_err);

        if (rel_err < 1e-3f || test_vals[i] == 0.0f) pass++;
        else tests_failed++;
    }
    printf("  %d/%d passed\n\n", pass, n);
    if (pass == n) tests_passed++;
}

static void test_ternary_dot_small(void) {
    printf("--- test_ternary_dot_small ---\n");

    /* Small test: 8 elements */
    float act_f[] = {1.0f, 2.0f, -3.0f, 0.5f, -0.5f, 4.0f, -1.0f, 0.0f};
    int8_t wt[] =   {  1,   -1,     1,    0,     1,   -1,     1,    0};
    int n = 8;

    /* Float reference */
    float ref = float_ternary_dot(act_f, wt, n);

    /* Convert to MTFP16 */
    mtfp16_t act_m16[8];
    for (int i = 0; i < n; i++) act_m16[i] = mtfp16_from_float(act_f[i]);

    /* Path A: per-element scalar */
    mtfp21_t result_a = mtfp_ternary_dot_scalar(act_m16, wt, n);
    float got_a = mtfp21_to_float(result_a);

    /* Path B: block exponent scalar */
    int16_t aligned_mant[8];
    int8_t block_exp = mtfp16_align_block(aligned_mant, act_m16, n);
    mtfp21_t result_b = mtfp_ternary_dot_block(aligned_mant, block_exp, wt, n);
    float got_b = mtfp21_to_float(result_b);

    /* Path B AVX2 */
    int16_t aligned_mant_padded[16] = {0};
    memcpy(aligned_mant_padded, aligned_mant, 8 * sizeof(int16_t));
    int8_t wt_padded[16] = {0};
    memcpy(wt_padded, wt, 8);
    mtfp21_t result_avx2 = mtfp_ternary_dot_avx2(aligned_mant_padded, block_exp, wt_padded, n);
    float got_avx2 = mtfp21_to_float(result_avx2);

    printf("  Float ref:    %.4f\n", ref);
    printf("  MTFP scalar:  %.4f (err=%.2e)\n", got_a, fabsf(got_a - ref));
    printf("  MTFP block:   %.4f (err=%.2e)\n", got_b, fabsf(got_b - ref));
    printf("  MTFP AVX2:    %.4f (err=%.2e)\n", got_avx2, fabsf(got_avx2 - ref));

    int pass = 0;
    if (fabsf(got_a - ref) < 0.01f) pass++;
    if (fabsf(got_b - ref) < 0.01f) pass++;
    if (fabsf(got_avx2 - ref) < 0.01f) pass++;
    printf("  %d/3 paths match reference\n\n", pass);
    if (pass == 3) tests_passed++; else tests_failed++;
}

static void test_ternary_dot_realistic(void) {
    printf("--- test_ternary_dot_realistic (n=2560) ---\n");

    int n = 2560;
    float * act_f = (float *)malloc(n * sizeof(float));
    int8_t * wt = (int8_t *)malloc(n * sizeof(int8_t));
    mtfp16_t * act_m16 = (mtfp16_t *)malloc(n * sizeof(mtfp16_t));
    int16_t * aligned = (int16_t *)calloc(n + 16, sizeof(int16_t));

    /* Generate realistic activation values (post-RMSNorm range) */
    srand(42);
    for (int i = 0; i < n; i++) {
        act_f[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;  /* [-1, 1] */
    }

    /* Generate ternary weights with ~33% sparsity */
    for (int i = 0; i < n; i++) {
        int r = rand() % 3;
        wt[i] = (int8_t)(r - 1);  /* {-1, 0, 1} */
    }

    /* Float reference */
    float ref = float_ternary_dot(act_f, wt, n);

    /* Convert to MTFP16 */
    for (int i = 0; i < n; i++) act_m16[i] = mtfp16_from_float(act_f[i]);

    /* Path A: per-element scalar */
    mtfp21_t result_a = mtfp_ternary_dot_scalar(act_m16, wt, n);
    float got_a = mtfp21_to_float(result_a);

    /* Path B: block exponent + AVX2 */
    int8_t block_exp = mtfp16_align_block(aligned, act_m16, n);
    mtfp21_t result_avx2 = mtfp_ternary_dot_avx2(aligned, block_exp, wt, n);
    float got_avx2 = mtfp21_to_float(result_avx2);

    float err_a = fabsf(ref) > 1e-6f ? fabsf((got_a - ref) / ref) : fabsf(got_a);
    float err_avx2 = fabsf(ref) > 1e-6f ? fabsf((got_avx2 - ref) / ref) : fabsf(got_avx2);

    printf("  Float ref:      %.6f\n", ref);
    printf("  MTFP scalar:    %.6f (rel_err=%.2e)\n", got_a, err_a);
    printf("  MTFP AVX2:      %.6f (rel_err=%.2e)\n", got_avx2, err_avx2);
    printf("  Block exponent: %d\n", block_exp);

    /* Check block alignment quality: how many mantissas were crushed to zero? */
    int zeros = 0;
    for (int i = 0; i < n; i++) if (aligned[i] == 0 && act_f[i] != 0.0f) zeros++;
    printf("  Block alignment: %d/%d values crushed to zero\n", zeros, n);

    int pass = 0;
    if (err_a < 1e-3f) { printf("  Scalar: PASS\n"); pass++; tests_passed++; }
    else { printf("  Scalar: FAIL\n"); tests_failed++; }
    if (err_avx2 < 0.01f) { printf("  AVX2:   PASS\n"); pass++; tests_passed++; }
    else { printf("  AVX2:   FAIL (may be block alignment loss)\n"); tests_failed++; }

    free(act_f); free(wt); free(act_m16); free(aligned);
    printf("\n");
}

static void test_ternary_dot_statistical(void) {
    printf("--- test_ternary_dot_statistical (1000 random vectors) ---\n");

    int n = 2560;
    float * act_f = (float *)malloc(n * sizeof(float));
    int8_t * wt = (int8_t *)malloc(n * sizeof(int8_t));
    mtfp16_t * act_m16 = (mtfp16_t *)malloc(n * sizeof(mtfp16_t));
    int16_t * aligned = (int16_t *)calloc(n + 16, sizeof(int16_t));

    int trials = 1000;
    float max_err_scalar = 0, max_err_avx2 = 0;
    double sum_err_scalar = 0, sum_err_avx2 = 0;

    for (int t = 0; t < trials; t++) {
        srand(t);
        for (int i = 0; i < n; i++) {
            act_f[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
        }
        for (int i = 0; i < n; i++) {
            wt[i] = (int8_t)((rand() % 3) - 1);
        }

        float ref = float_ternary_dot(act_f, wt, n);
        for (int i = 0; i < n; i++) act_m16[i] = mtfp16_from_float(act_f[i]);

        mtfp21_t result_a = mtfp_ternary_dot_scalar(act_m16, wt, n);
        float got_a = mtfp21_to_float(result_a);

        int8_t block_exp = mtfp16_align_block(aligned, act_m16, n);
        mtfp21_t result_avx2 = mtfp_ternary_dot_avx2(aligned, block_exp, wt, n);
        float got_avx2 = mtfp21_to_float(result_avx2);

        /* Use absolute error when reference is small — relative error
         * is meaningless when the dot product is near cancellation */
        float err_a = fabsf(ref) > 0.1f ? fabsf((got_a - ref) / ref) : fabsf(got_a - ref);
        float err_avx2 = fabsf(ref) > 0.1f ? fabsf((got_avx2 - ref) / ref) : fabsf(got_avx2 - ref);

        if (err_a > max_err_scalar) max_err_scalar = err_a;
        if (err_avx2 > max_err_avx2) max_err_avx2 = err_avx2;
        sum_err_scalar += err_a;
        sum_err_avx2 += err_avx2;
    }

    printf("  Scalar:  avg_err=%.2e  max_err=%.2e\n",
           sum_err_scalar / trials, max_err_scalar);
    printf("  AVX2:    avg_err=%.2e  max_err=%.2e\n",
           sum_err_avx2 / trials, max_err_avx2);

    /* Threshold: 1% max error across 1000 random dot products.
     * The error comes from MTFP16's 10-trit mantissa (~15.8 bits)
     * vs float32's 24-bit mantissa. Sub-1% is expected. */
    if (max_err_scalar < 0.01f) { printf("  Scalar: PASS\n"); tests_passed++; }
    else { printf("  Scalar: FAIL\n"); tests_failed++; }
    if (max_err_avx2 < 0.01f) { printf("  AVX2:   PASS\n"); tests_passed++; }
    else { printf("  AVX2:   FAIL\n"); tests_failed++; }

    free(act_f); free(wt); free(act_m16); free(aligned);
    printf("\n");
}

/* ================================================================
 *  Main
 * ================================================================ */

int main(void) {
    printf("MTFP16 Ternary Matmul Validation\n");
    printf("================================\n\n");

    test_mtfp16_conversion();
    test_ternary_dot_small();
    test_ternary_dot_realistic();
    test_ternary_dot_statistical();

    printf("================================\n");
    printf("Results: %d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
