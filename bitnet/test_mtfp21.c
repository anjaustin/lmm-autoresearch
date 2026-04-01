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

    printf("======================\n");
    printf("Results: %d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
