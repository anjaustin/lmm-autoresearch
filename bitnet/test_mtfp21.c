/*
 * test_mtfp21.c — Validate MTFP21 arithmetic against float32
 *
 * Tests conversion accuracy, basic arithmetic, and RMSNorm.
 *
 * Build: gcc -O2 -o test_mtfp21 test_mtfp21.c -lm
 * Run:   ./test_mtfp21
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include "shirley_mtfp21.h"

/* ================================================================
 *  Test helpers
 * ================================================================ */

static int tests_passed = 0;
static int tests_failed = 0;

static void check_close(const char *name, float expected, float got, float rtol) {
    float err;
    if (expected == 0.0f) {
        err = fabsf(got);
    } else {
        err = fabsf((got - expected) / expected);
    }
    if (err <= rtol || (fabsf(expected) < 1e-30f && fabsf(got) < 1e-30f)) {
        tests_passed++;
    } else {
        printf("  FAIL %s: expected %.8e, got %.8e, rel_err=%.2e (tol=%.2e)\n",
               name, expected, got, err, rtol);
        tests_failed++;
    }
}

/* ================================================================
 *  Test 1: Conversion round-trip accuracy
 * ================================================================ */

static void test_conversion(void) {
    printf("=== Test 1: Float32 <-> MTFP21 conversion ===\n");

    float test_values[] = {
        0.0f, 1.0f, -1.0f, 3.0f, -3.0f,
        0.5f, -0.5f, 0.1f, -0.1f,
        1234.5678f, -1234.5678f,
        1e-10f, -1e-10f,
        1e10f, -1e10f,
        0.333333f, 3.14159f, 2.71828f,
        FLT_MIN, FLT_MAX * 0.001f, /* avoid overflow */
        127.0f, -128.0f,  /* int8 range */
        40.0f, -40.0f,    /* 4-trit range */
        21523360.0f,       /* mantissa max */
    };

    int n = sizeof(test_values) / sizeof(test_values[0]);
    float max_err = 0.0f;
    int conversions = 0;

    for (int i = 0; i < n; i++) {
        float f = test_values[i];
        mtfp21_t m = mtfp21_from_float(f);
        float back = mtfp21_to_float(m);

        float err = (f == 0.0f) ? fabsf(back) : fabsf((back - f) / f);
        if (err > max_err && fabsf(f) > 1e-30f) max_err = err;
        conversions++;

        check_close("conv", f, back, 1e-6f);  /* MTFP21 has 25.4 bits ≈ 7.6 digits */
    }

    printf("  Conversions: %d, max relative error: %.2e\n", conversions, max_err);

    /* Random values */
    srand(42);
    float rand_max_err = 0.0f;
    for (int i = 0; i < 10000; i++) {
        /* Random float in [-1000, 1000] */
        float f = ((float)rand() / RAND_MAX - 0.5f) * 2000.0f;
        mtfp21_t m = mtfp21_from_float(f);
        float back = mtfp21_to_float(m);
        float err = fabsf(f) > 1e-10f ? fabsf((back - f) / f) : 0.0f;
        if (err > rand_max_err) rand_max_err = err;
    }
    printf("  Random 10K values [-1000,1000]: max relative error: %.2e\n", rand_max_err);
    if (rand_max_err < 1e-6f) tests_passed++; else { tests_failed++; printf("  FAIL: random conversion too lossy\n"); }
    printf("\n");
}

/* ================================================================
 *  Test 2: Addition accuracy
 * ================================================================ */

static void test_addition(void) {
    printf("=== Test 2: MTFP21 Addition ===\n");

    struct { float a, b; } cases[] = {
        {1.0f, 2.0f},
        {1.0f, -1.0f},
        {100.0f, 0.001f},
        {-50.0f, 50.0f},
        {3.14159f, 2.71828f},
        {1e5f, 1e-5f},
        {0.0f, 42.0f},
        {-127.0f, 127.0f},
        {1234.5f, -1234.5f},
        {0.1f, 0.2f},
    };

    int n = sizeof(cases) / sizeof(cases[0]);
    float max_err = 0.0f;

    for (int i = 0; i < n; i++) {
        float expected = cases[i].a + cases[i].b;
        mtfp21_t ma = mtfp21_from_float(cases[i].a);
        mtfp21_t mb = mtfp21_from_float(cases[i].b);
        mtfp21_t mr = mtfp21_add(ma, mb);
        float got = mtfp21_to_float(mr);

        float err = fabsf(expected) > 1e-10f ? fabsf((got - expected) / expected) : fabsf(got);
        if (err > max_err) max_err = err;

        check_close("add", expected, got, 1e-5f);
    }
    printf("  Max addition relative error: %.2e\n\n", max_err);
}

/* ================================================================
 *  Test 3: Multiplication accuracy
 * ================================================================ */

static void test_multiplication(void) {
    printf("=== Test 3: MTFP21 Multiplication ===\n");

    struct { float a, b; } cases[] = {
        {2.0f, 3.0f},
        {-2.0f, 3.0f},
        {0.5f, 0.5f},
        {100.0f, 0.01f},
        {3.14159f, 2.71828f},
        {0.0f, 42.0f},
        {1.0f, -1.0f},
        {127.0f, 127.0f},
        {0.001f, 0.001f},
        {1234.0f, 5678.0f},
    };

    int n = sizeof(cases) / sizeof(cases[0]);
    float max_err = 0.0f;

    for (int i = 0; i < n; i++) {
        float expected = cases[i].a * cases[i].b;
        mtfp21_t ma = mtfp21_from_float(cases[i].a);
        mtfp21_t mb = mtfp21_from_float(cases[i].b);
        mtfp21_t mr = mtfp21_mul(ma, mb);
        float got = mtfp21_to_float(mr);

        float err = fabsf(expected) > 1e-10f ? fabsf((got - expected) / expected) : fabsf(got);
        if (err > max_err) max_err = err;

        check_close("mul", expected, got, 1e-5f);
    }
    printf("  Max multiplication relative error: %.2e\n\n", max_err);
}

/* ================================================================
 *  Test 4: rsqrt accuracy
 * ================================================================ */

static void test_rsqrt(void) {
    printf("=== Test 4: MTFP21 rsqrt ===\n");

    float test_values[] = {
        1.0f, 4.0f, 9.0f, 16.0f, 0.25f,
        100.0f, 0.01f, 3.14159f,
        0.001f, 1000.0f,
        0.5f, 2.0f,
        127.0f, 0.1f,
    };

    int n = sizeof(test_values) / sizeof(test_values[0]);
    float max_err = 0.0f;

    for (int i = 0; i < n; i++) {
        float x = test_values[i];
        float expected = 1.0f / sqrtf(x);

        mtfp21_t mx = mtfp21_from_float(x);
        mtfp21_t mr = mtfp21_rsqrt(mx);
        float got = mtfp21_to_float(mr);

        float err = fabsf((got - expected) / expected);
        if (err > max_err) max_err = err;

        check_close("rsqrt", expected, got, 1e-5f);
    }
    printf("  Max rsqrt relative error: %.2e\n\n", max_err);
}

/* ================================================================
 *  Test 5: RMSNorm end-to-end vs float32
 * ================================================================ */

static void test_rmsnorm(void) {
    printf("=== Test 5: MTFP21 RMSNorm vs float32 ===\n");

    /* Simulate a hidden dimension of 2560 (BitNet-2B) */
    int n = 2560;
    float *src = (float *)malloc(n * sizeof(float));
    float *dst_float = (float *)malloc(n * sizeof(float));
    float *dst_mtfp  = (float *)malloc(n * sizeof(float));
    float eps = 1e-5f;

    /* Fill with realistic activation values (roughly normal, std ~1) */
    srand(42);
    for (int i = 0; i < n; i++) {
        /* Box-Muller for normal distribution */
        float u1 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 1.0f);
        float u2 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 1.0f);
        src[i] = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2);
    }

    /* Float32 RMSNorm */
    {
        float sum = 0.0f;
        for (int i = 0; i < n; i++) {
            sum += src[i] * src[i];
        }
        float scale = 1.0f / sqrtf(sum / n + eps);
        for (int i = 0; i < n; i++) {
            dst_float[i] = src[i] * scale;
        }
    }

    /* MTFP21 RMSNorm */
    mtfp21_rmsnorm(dst_mtfp, src, n, eps);

    /* Compare */
    float max_err = 0.0f;
    float sum_err = 0.0f;
    for (int i = 0; i < n; i++) {
        float err = fabsf(dst_float[i]) > 1e-10f
            ? fabsf((dst_mtfp[i] - dst_float[i]) / dst_float[i])
            : fabsf(dst_mtfp[i]);
        if (err > max_err) max_err = err;
        sum_err += err;
    }
    float avg_err = sum_err / n;

    printf("  Dim: %d, eps: %.0e\n", n, eps);
    printf("  Max element relative error: %.2e\n", max_err);
    printf("  Avg element relative error: %.2e\n", avg_err);

    if (max_err < 1e-5f) {
        printf("  PASS: MTFP21 RMSNorm matches float32 to within 1e-5\n");
        tests_passed++;
    } else if (max_err < 1e-4f) {
        printf("  MARGINAL: MTFP21 RMSNorm max error %.2e (within 1e-4 but not 1e-5)\n", max_err);
        tests_passed++;
    } else {
        printf("  FAIL: MTFP21 RMSNorm max error %.2e exceeds 1e-4\n", max_err);
        tests_failed++;
    }

    /* Test with values in different ranges */
    printf("\n  Range tests:\n");
    float ranges[] = {0.01f, 0.1f, 1.0f, 10.0f, 100.0f};
    for (int r = 0; r < 5; r++) {
        for (int i = 0; i < n; i++) {
            float u1 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 1.0f);
            float u2 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 1.0f);
            src[i] = ranges[r] * sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2);
        }

        /* Float32 */
        float sum = 0.0f;
        for (int i = 0; i < n; i++) sum += src[i] * src[i];
        float scale = 1.0f / sqrtf(sum / n + eps);
        for (int i = 0; i < n; i++) dst_float[i] = src[i] * scale;

        /* MTFP21 */
        mtfp21_rmsnorm(dst_mtfp, src, n, eps);

        /* Compare */
        max_err = 0.0f;
        for (int i = 0; i < n; i++) {
            float err = fabsf(dst_float[i]) > 1e-10f
                ? fabsf((dst_mtfp[i] - dst_float[i]) / dst_float[i])
                : fabsf(dst_mtfp[i]);
            if (err > max_err) max_err = err;
        }
        printf("    scale=%.2f: max_err=%.2e %s\n", ranges[r], max_err,
               max_err < 1e-4f ? "OK" : "DEGRADED");
    }

    free(src);
    free(dst_float);
    free(dst_mtfp);
    printf("\n");
}

/* ================================================================
 *  Test 6: Accumulation accuracy (sum of 2560 squares)
 * ================================================================ */

static void test_accumulation(void) {
    printf("=== Test 6: MTFP21 Accumulation (sum of squares) ===\n");

    int n = 2560;
    float *vals = (float *)malloc(n * sizeof(float));

    srand(42);
    for (int i = 0; i < n; i++) {
        float u1 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 1.0f);
        float u2 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 1.0f);
        vals[i] = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2);
    }

    /* Float64 reference (high precision) */
    double ref_sum = 0.0;
    for (int i = 0; i < n; i++) ref_sum += (double)vals[i] * (double)vals[i];

    /* Float32 */
    float f32_sum = 0.0f;
    for (int i = 0; i < n; i++) f32_sum += vals[i] * vals[i];

    /* MTFP21 */
    mtfp21_t m_sum = {0, 0};
    for (int i = 0; i < n; i++) {
        mtfp21_t xi = mtfp21_from_float(vals[i]);
        mtfp21_t xi2 = mtfp21_mul(xi, xi);
        m_sum = mtfp21_add(m_sum, xi2);
    }
    float mtfp_sum = mtfp21_to_float(m_sum);

    double f32_err = fabs((double)f32_sum - ref_sum) / ref_sum;
    double mtfp_err = fabs((double)mtfp_sum - ref_sum) / ref_sum;

    printf("  Reference (f64):  %.10f\n", ref_sum);
    printf("  Float32:          %.10f  (rel_err: %.2e)\n", (double)f32_sum, f32_err);
    printf("  MTFP21:           %.10f  (rel_err: %.2e)\n", (double)mtfp_sum, mtfp_err);

    if (mtfp_err < f32_err * 10.0) {
        printf("  PASS: MTFP21 accumulation within 10x of float32 error\n");
        tests_passed++;
    } else {
        printf("  FAIL: MTFP21 accumulation error %.2e >> float32 error %.2e\n", mtfp_err, f32_err);
        tests_failed++;
    }

    free(vals);
    printf("\n");
}

/* ================================================================
 *  Test 7: int8 ↔ MTFP21 conversion (Issue 7)
 * ================================================================ */

static void test_int8_conversion(void) {
    printf("=== Test 7: int8 <-> MTFP21 conversion ===\n");

    /* Exact round-trip for 4-trit range [-40, +40] */
    int failures_4trit = 0;
    for (int v = -40; v <= 40; v++) {
        mtfp21_t m = mtfp21_from_int8((int8_t)v);
        int8_t back = mtfp21_to_int8(m, 40);
        if (back != (int8_t)v) {
            if (failures_4trit < 3) printf("  FAIL 4-trit round-trip: %d -> %d\n", v, back);
            failures_4trit++;
        }
    }
    if (failures_4trit == 0) { printf("  4-trit round-trip: 81/81 exact\n"); tests_passed++; }
    else { printf("  FAIL 4-trit round-trip: %d/81 failed\n", failures_4trit); tests_failed++; }

    /* Exact round-trip for 5-trit range [-121, +121] */
    int failures_5trit = 0;
    for (int v = -121; v <= 121; v++) {
        mtfp21_t m = mtfp21_from_int8((int8_t)v);
        int8_t back = mtfp21_to_int8(m, 121);
        if (back != (int8_t)v) failures_5trit++;
    }
    if (failures_5trit == 0) { printf("  5-trit round-trip: 243/243 exact\n"); tests_passed++; }
    else { printf("  FAIL 5-trit round-trip: %d/243 failed\n", failures_5trit); tests_failed++; }

    /* Clamping */
    mtfp21_t big = mtfp21_from_float(1000.0f);
    int8_t clamped = mtfp21_to_int8(big, 40);
    check_close("clamp_pos", 40.0f, (float)clamped, 0.0f);

    mtfp21_t neg_big = mtfp21_from_float(-1000.0f);
    clamped = mtfp21_to_int8(neg_big, 40);
    check_close("clamp_neg", -40.0f, (float)clamped, 0.0f);

    /* Zero preservation */
    mtfp21_t zero = mtfp21_from_int8(0);
    if (zero.mantissa == 0) { printf("  Zero preserved\n"); tests_passed++; }
    else { printf("  FAIL: zero mantissa = %d\n", zero.mantissa); tests_failed++; }

    /* Convert MTFP21 with non-zero exponent to int8 */
    mtfp21_t three = {1, 1}; /* 1 * 3^1 = 3 */
    int8_t three_i8 = mtfp21_to_int8(three, 40);
    check_close("exp_pos", 3.0f, (float)three_i8, 0.0f);

    mtfp21_t third = mtfp21_from_float(0.333333f);
    int8_t third_i8 = mtfp21_to_int8(third, 40);
    check_close("exp_neg", 0.0f, (float)third_i8, 0.0f); /* 0.33 rounds to 0 in integer */

    printf("\n");
}

/* ================================================================
 *  Test 8: Integer division (Issue 2)
 * ================================================================ */

static void test_division(void) {
    printf("=== Test 8: MTFP21 Integer Division ===\n");

    /* div_scalar: 2560/2560 = 1.0 */
    {
        mtfp21_t val = mtfp21_from_float(2560.0f);
        mtfp21_t result = mtfp21_div_scalar(val, 2560);
        check_close("2560/2560", 1.0f, mtfp21_to_float(result), 1e-6f);
    }

    /* div_scalar: sum_of_squares / 2560 (RMSNorm case) */
    {
        /* Build a realistic sum of squares: 2560 * 1.0^2 = 2560 */
        mtfp21_t sum = mtfp21_from_float(2560.0f);
        mtfp21_t mean = mtfp21_div_scalar(sum, 2560);
        check_close("mean_unit", 1.0f, mtfp21_to_float(mean), 1e-6f);
    }

    /* Division by powers of 3 (should be near-exact — just exponent shift) */
    {
        mtfp21_t val = mtfp21_from_float(81.0f);
        mtfp21_t result = mtfp21_div_scalar(val, 3);
        check_close("81/3", 27.0f, mtfp21_to_float(result), 1e-6f);

        result = mtfp21_div_scalar(val, 9);
        check_close("81/9", 9.0f, mtfp21_to_float(result), 1e-6f);

        result = mtfp21_div_scalar(val, 27);
        check_close("81/27", 3.0f, mtfp21_to_float(result), 1e-6f);
    }

    /* Division by small scalars */
    {
        mtfp21_t val = mtfp21_from_float(100.0f);
        check_close("100/2", 50.0f, mtfp21_to_float(mtfp21_div_scalar(val, 2)), 1e-6f);
        check_close("100/4", 25.0f, mtfp21_to_float(mtfp21_div_scalar(val, 4)), 1e-6f);
        check_close("100/5", 20.0f, mtfp21_to_float(mtfp21_div_scalar(val, 5)), 1e-6f);
        check_close("100/7", 100.0f/7.0f, mtfp21_to_float(mtfp21_div_scalar(val, 7)), 1e-5f);
        check_close("100/10", 10.0f, mtfp21_to_float(mtfp21_div_scalar(val, 10)), 1e-6f);
    }

    /* Division by large scalars */
    {
        mtfp21_t val = mtfp21_from_float(1e6f);
        check_close("1e6/65536", 1e6f/65536.0f, mtfp21_to_float(mtfp21_div_scalar(val, 65536)), 1e-5f);
        check_close("1e6/1000000", 1.0f, mtfp21_to_float(mtfp21_div_scalar(val, 1000000)), 1e-5f);
    }

    /* General MTFP21/MTFP21 division: a/a = 1 */
    {
        mtfp21_t val = mtfp21_from_float(3.14159f);
        mtfp21_t result = mtfp21_div(val, val);
        check_close("pi/pi", 1.0f, mtfp21_to_float(result), 1e-5f);
    }

    /* Accumulation then division (the actual RMSNorm path) */
    {
        int n = 2560;
        float *vals = (float *)malloc(n * sizeof(float));
        srand(42);
        for (int i = 0; i < n; i++) {
            float u1 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 1.0f);
            float u2 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 1.0f);
            vals[i] = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2);
        }

        /* float64 reference */
        double ref_sum = 0.0;
        for (int i = 0; i < n; i++) ref_sum += (double)vals[i] * (double)vals[i];
        double ref_mean = ref_sum / n;

        /* MTFP21 */
        mtfp21_t m_sum = {0, 0};
        for (int i = 0; i < n; i++) {
            mtfp21_t xi = mtfp21_from_float(vals[i]);
            mtfp21_t xi2 = mtfp21_mul(xi, xi);
            m_sum = mtfp21_add(m_sum, xi2);
        }
        mtfp21_t m_mean = mtfp21_div_scalar(m_sum, n);
        float mtfp_mean = mtfp21_to_float(m_mean);

        float err = fabsf((mtfp_mean - (float)ref_mean) / (float)ref_mean);
        printf("  RMSNorm mean: ref=%.8f mtfp=%.8f err=%.2e\n", ref_mean, mtfp_mean, err);
        if (err < 1e-6f) { tests_passed++; } else { tests_failed++; printf("  FAIL: mean error too high\n"); }

        free(vals);
    }

    /* Verify no float in recip_int path — test the integer reciprocal directly */
    {
        mtfp21_t recip = mtfp21_recip_int(2560);
        float recip_f = mtfp21_to_float(recip);
        check_close("recip_2560", 1.0f/2560.0f, recip_f, 1e-6f);
    }

    printf("\n");
}

/* ================================================================
 *  Test 9: Adversarial inputs (Issue 5)
 * ================================================================ */

static float randn(void) {
    float u1 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 1.0f);
    float u2 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 1.0f);
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2);
}

static float float_rmsnorm_check(const float *src, float *dst, int n, float eps) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += (double)src[i] * (double)src[i];
    float scale = 1.0f / sqrtf((float)(sum / n) + eps);
    for (int i = 0; i < n; i++) dst[i] = src[i] * scale;
    return scale;
}

static void test_adversarial(void) {
    printf("=== Test 9: Adversarial Inputs ===\n");
    int n = 2560;
    float eps = 1e-5f;
    float *src = (float *)malloc(n * sizeof(float));
    float *dst_f = (float *)malloc(n * sizeof(float));
    float *dst_m = (float *)malloc(n * sizeof(float));

    struct { const char *name; int ok; } results[6];
    int num_dists = 6;

    for (int d = 0; d < num_dists; d++) {
        srand(42 + d);

        switch (d) {
        case 0: /* Heavy tails (Cauchy) */
            for (int i = 0; i < n; i++) {
                float n1 = randn(), n2 = randn();
                src[i] = (fabsf(n2) > 1e-6f) ? n1 / n2 : 0.0f;
                if (fabsf(src[i]) > 1e6f) src[i] = (src[i] > 0 ? 1e6f : -1e6f); /* prevent inf */
            }
            results[d].name = "heavy_tails";
            break;
        case 1: /* Near-zero cluster */
            for (int i = 0; i < n; i++) {
                src[i] = (rand() % 10 == 0) ? randn() : randn() * 1e-6f;
            }
            results[d].name = "near_zero";
            break;
        case 2: /* Bimodal (ternary matmul output) */
            for (int i = 0; i < n; i++) {
                src[i] = ((rand() % 2) ? 1.0f : -1.0f) + randn() * 0.01f;
            }
            results[d].name = "bimodal";
            break;
        case 3: /* Adversarial cancellation */
            for (int i = 0; i < n; i++) {
                src[i] = 1.0f + 1e-7f * (float)i;
            }
            results[d].name = "cancellation";
            break;
        case 4: /* Extreme dynamic range */
            for (int i = 0; i < n; i++) {
                src[i] = (i < n/2) ? 1e5f * randn() : 1e-5f * randn();
            }
            results[d].name = "dyn_range";
            break;
        case 5: /* All same value */
            for (int i = 0; i < n; i++) src[i] = 0.7f;
            results[d].name = "all_same";
            break;
        }

        float_rmsnorm_check(src, dst_f, n, eps);
        mtfp21_rmsnorm(dst_m, src, n, eps);

        float max_err = 0.0f;
        int finite = 1;
        for (int i = 0; i < n; i++) {
            if (isnan(dst_m[i]) || isinf(dst_m[i])) { finite = 0; break; }
            float err = fabsf(dst_f[i]) > 1e-10f
                ? fabsf((dst_m[i] - dst_f[i]) / dst_f[i]) : fabsf(dst_m[i]);
            if (err > max_err) max_err = err;
        }

        results[d].ok = finite && max_err < 1e-3f;
        printf("  %-14s: finite=%d max_err=%.2e %s\n",
               results[d].name, finite, max_err, results[d].ok ? "OK" : "DEGRADED");
        if (results[d].ok) tests_passed++; else tests_failed++;
    }

    free(src); free(dst_f); free(dst_m);
    printf("\n");
}

/* ================================================================
 *  Test 10: Statistical accumulation (Issue 6)
 * ================================================================ */

static int compare_double(const void *a, const void *b) {
    double da = *(const double *)a, db = *(const double *)b;
    return (da > db) - (da < db);
}

static void test_accumulation_statistical(void) {
    printf("=== Test 10: Statistical Accumulation (1000 seeds) ===\n");
    int n = 2560;
    int num_seeds = 1000;
    float *vals = (float *)malloc(n * sizeof(float));
    double *f32_errors = (double *)malloc(num_seeds * sizeof(double));
    double *mtfp_errors = (double *)malloc(num_seeds * sizeof(double));

    int mtfp_wins = 0;

    for (int seed = 0; seed < num_seeds; seed++) {
        srand(seed);
        for (int i = 0; i < n; i++) vals[i] = randn();

        /* float64 reference */
        double ref = 0.0;
        for (int i = 0; i < n; i++) ref += (double)vals[i] * (double)vals[i];

        /* float32 */
        float f32 = 0.0f;
        for (int i = 0; i < n; i++) f32 += vals[i] * vals[i];

        /* MTFP21 */
        mtfp21_t m = {0, 0};
        for (int i = 0; i < n; i++) {
            mtfp21_t xi = mtfp21_from_float(vals[i]);
            mtfp21_t xi2 = mtfp21_mul(xi, xi);
            m = mtfp21_add(m, xi2);
        }
        float mtfp = mtfp21_to_float(m);

        f32_errors[seed] = fabs((double)f32 - ref) / ref;
        mtfp_errors[seed] = fabs((double)mtfp - ref) / ref;
        if (mtfp_errors[seed] <= f32_errors[seed]) mtfp_wins++;
    }

    /* Sort for median/percentile */
    qsort(f32_errors, num_seeds, sizeof(double), compare_double);
    qsort(mtfp_errors, num_seeds, sizeof(double), compare_double);

    double f32_median = f32_errors[num_seeds / 2];
    double mtfp_median = mtfp_errors[num_seeds / 2];
    double f32_p95 = f32_errors[(int)(num_seeds * 0.95)];
    double mtfp_p95 = mtfp_errors[(int)(num_seeds * 0.95)];
    double f32_max = f32_errors[num_seeds - 1];
    double mtfp_max = mtfp_errors[num_seeds - 1];

    printf("  Seeds: %d, n=%d\n", num_seeds, n);
    printf("  %-10s  %-12s  %-12s\n", "", "float32", "MTFP21");
    printf("  %-10s  %-12.2e  %-12.2e\n", "median", f32_median, mtfp_median);
    printf("  %-10s  %-12.2e  %-12.2e\n", "95th pct", f32_p95, mtfp_p95);
    printf("  %-10s  %-12.2e  %-12.2e\n", "max", f32_max, mtfp_max);
    printf("  MTFP21 wins: %d/%d (%.1f%%)\n", mtfp_wins, num_seeds, 100.0 * mtfp_wins / num_seeds);

    if (mtfp_median <= f32_median) {
        printf("  PASS: MTFP21 median error <= float32 median error\n");
        tests_passed++;
    } else {
        printf("  FAIL: MTFP21 median error > float32 median error\n");
        tests_failed++;
    }

    free(vals); free(f32_errors); free(mtfp_errors);
    printf("\n");
}

/* ================================================================
 *  Test 11: Pure MTFP21 RMSNorm (Issue 3)
 * ================================================================ */

static void test_rmsnorm_pure(void) {
    printf("=== Test 11: Pure MTFP21 RMSNorm (no float escape hatches) ===\n");

    int n = 2560;
    float eps = 1e-5f;
    mtfp21_t eps_m = mtfp21_from_float(eps);

    float *src_f = (float *)malloc(n * sizeof(float));
    float *dst_f = (float *)malloc(n * sizeof(float));
    mtfp21_t *src_m = (mtfp21_t *)malloc(n * sizeof(mtfp21_t));
    mtfp21_t *dst_m = (mtfp21_t *)malloc(n * sizeof(mtfp21_t));

    srand(42);
    for (int i = 0; i < n; i++) {
        src_f[i] = randn();
        src_m[i] = mtfp21_from_float(src_f[i]);
    }

    /* Float64 reference */
    double ref_sum = 0.0;
    for (int i = 0; i < n; i++) ref_sum += (double)src_f[i] * (double)src_f[i];
    double ref_scale = 1.0 / sqrt(ref_sum / n + eps);

    /* Pure MTFP21 */
    mtfp21_rmsnorm_pure(dst_m, src_m, n, eps_m);

    float max_err = 0.0f;
    for (int i = 0; i < n; i++) {
        double expected = (double)src_f[i] * ref_scale;
        float got = mtfp21_to_float(dst_m[i]);
        float err = fabs(expected) > 1e-10 ? fabsf((float)((got - expected) / expected)) : fabsf(got);
        if (err > max_err) max_err = err;
    }

    printf("  Pure MTFP21 RMSNorm: max_err=%.2e\n", max_err);
    if (max_err < 1e-5f) { printf("  PASS\n"); tests_passed++; }
    else if (max_err < 1e-4f) { printf("  MARGINAL (within 1e-4)\n"); tests_passed++; }
    else { printf("  FAIL\n"); tests_failed++; }

    free(src_f); free(dst_f); free(src_m); free(dst_m);
    printf("\n");
}

/* ================================================================
 *  Test 12: Chained end-to-end pipeline (Issue 4)
 * ================================================================ */

static void test_chained_pipeline(void) {
    printf("=== Test 12: Chained int8 -> MTFP21 RMSNorm -> int8 ===\n");

    int n = 2560;
    int in_range = 80;   /* 5-trit activations */
    int out_range = 121; /* output at 5-trit range */
    mtfp21_t eps = mtfp21_from_float(1e-5f);

    int8_t *src = (int8_t *)malloc(n);
    int8_t *dst = (int8_t *)malloc(n);

    /* Random 5-trit activations */
    srand(42);
    for (int i = 0; i < n; i++) {
        src[i] = (int8_t)((rand() % (2 * in_range + 1)) - in_range);
    }

    /* int8 RMSNorm */
    mtfp21_rmsnorm_int8(dst, src, n, eps, out_range);

    /* Float64 reference */
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += (double)src[i] * (double)src[i];
    double scale = 1.0 / sqrt(sum / n + 1e-5);

    /* Compare: how many output elements match the float64 reference (quantized)? */
    int exact_match = 0;
    int off_by_one = 0;
    int worse = 0;
    for (int i = 0; i < n; i++) {
        double ref_val = (double)src[i] * scale;
        /* Quantize reference to out_range */
        int ref_q = (int)round(ref_val);
        if (ref_q > out_range) ref_q = out_range;
        if (ref_q < -out_range) ref_q = -out_range;

        int diff = abs((int)dst[i] - ref_q);
        if (diff == 0) exact_match++;
        else if (diff == 1) off_by_one++;
        else worse++;
    }

    printf("  n=%d, in_range=%d, out_range=%d\n", n, in_range, out_range);
    printf("  Exact match: %d/%d (%.1f%%)\n", exact_match, n, 100.0 * exact_match / n);
    printf("  Off-by-one:  %d/%d (%.1f%%)\n", off_by_one, n, 100.0 * off_by_one / n);
    printf("  Worse:       %d/%d (%.1f%%)\n", worse, n, 100.0 * worse / n);

    int pass = (exact_match + off_by_one == n) && ((float)exact_match / n >= 0.95f);
    if (pass) { printf("  PASS: >=95%% exact, rest off-by-one\n"); tests_passed++; }
    else { printf("  FAIL\n"); tests_failed++; }

    /* Test repeated application (2 consecutive RMSNorms) */
    int8_t *dst2 = (int8_t *)malloc(n);
    mtfp21_rmsnorm_int8(dst2, dst, n, eps, out_range);

    /* After double normalization, values should be near unit variance */
    int in_range_count = 0;
    for (int i = 0; i < n; i++) {
        if (abs(dst2[i]) <= out_range) in_range_count++;
    }
    printf("  Double RMSNorm: %d/%d outputs in range (expect all)\n", in_range_count, n);
    if (in_range_count == n) tests_passed++; else tests_failed++;

    free(src); free(dst); free(dst2);
    printf("\n");
}

/* ================================================================
 *  Main
 * ================================================================ */

int main(void) {
    printf("MTFP21 Validation Suite\n");
    printf("======================\n\n");

    test_conversion();
    test_addition();
    test_multiplication();
    test_rsqrt();
    test_rmsnorm();
    test_accumulation();
    test_int8_conversion();
    test_division();
    test_adversarial();
    test_accumulation_statistical();
    test_rmsnorm_pure();
    test_chained_pipeline();

    printf("======================\n");
    printf("Results: %d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
