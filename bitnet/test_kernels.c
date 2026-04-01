/*
 * test_kernels.c — Validate and benchmark AVX2 integer RMSNorm kernel
 *
 * Build: gcc -O2 -mavx2 -march=native -o test_kernels test_kernels.c -lm
 * Run:   ./test_kernels
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include "shirley_kernels.h"
#include "shirley_mtfp21.h"

static int tests_passed = 0;
static int tests_failed = 0;

static float randn(void) {
    float u1 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 1.0f);
    float u2 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 1.0f);
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2);
}

static double elapsed_us(struct timespec t0, struct timespec t1) {
    return (t1.tv_sec - t0.tv_sec) * 1e6 + (t1.tv_nsec - t0.tv_nsec) / 1e3;
}

/* ================================================================
 *  Test 1: Correctness against float64 reference
 * ================================================================ */

static void test_correctness(void) {
    printf("=== Test 1: AVX2 RMSNorm correctness ===\n");

    int n = 2560;
    int in_range = 80;
    int out_range = 80;
    float eps = 1e-5f;

    int8_t *src = (int8_t *)malloc(n);
    int8_t *dst = (int8_t *)malloc(n);

    srand(42);
    for (int i = 0; i < n; i++) {
        src[i] = (int8_t)((rand() % (2 * in_range + 1)) - in_range);
    }

    /* AVX2 kernel */
    ternary_rmsnorm_avx2(dst, src, n, eps, out_range);

    /* Float64 reference */
    double sum_sq = 0.0;
    for (int i = 0; i < n; i++) sum_sq += (double)src[i] * (double)src[i];
    double scale = 1.0 / sqrt(sum_sq / n + eps);

    int exact = 0, off1 = 0, worse = 0;
    for (int i = 0; i < n; i++) {
        int ref_q = (int)round((double)src[i] * scale);
        if (ref_q > out_range) ref_q = out_range;
        if (ref_q < -out_range) ref_q = -out_range;

        int diff = abs((int)dst[i] - ref_q);
        if (diff == 0) exact++;
        else if (diff == 1) off1++;
        else worse++;
    }

    printf("  Exact: %d/%d (%.1f%%), Off-by-1: %d, Worse: %d\n",
           exact, n, 100.0 * exact / n, off1, worse);

    if (worse == 0 && (float)exact / n >= 0.95f) {
        printf("  PASS\n"); tests_passed++;
    } else {
        printf("  FAIL\n"); tests_failed++;
    }
}

/* ================================================================
 *  Test 2: Multiple dimensions and ranges
 * ================================================================ */

static void test_dimensions(void) {
    printf("=== Test 2: Multiple dimensions and ranges ===\n");

    int dims[] = {64, 256, 640, 2560, 5120};
    int ranges[] = {40, 80, 121};

    for (int d = 0; d < 5; d++) {
        for (int r = 0; r < 3; r++) {
            int n = dims[d];
            int out_range = ranges[r];

            int8_t *src = (int8_t *)malloc(n);
            int8_t *dst = (int8_t *)malloc(n);

            srand(42 + d * 10 + r);
            for (int i = 0; i < n; i++) {
                src[i] = (int8_t)((rand() % (2 * out_range + 1)) - out_range);
            }

            ternary_rmsnorm_avx2(dst, src, n, 1e-5f, out_range);

            /* Check all outputs in range */
            int ok = 1;
            for (int i = 0; i < n; i++) {
                if (dst[i] > out_range || dst[i] < -out_range) { ok = 0; break; }
            }

            /* Check against reference */
            double sum_sq = 0.0;
            for (int i = 0; i < n; i++) sum_sq += (double)src[i] * (double)src[i];
            double scale = 1.0 / sqrt(sum_sq / n + 1e-5);

            int worse = 0;
            for (int i = 0; i < n; i++) {
                int ref_q = (int)round((double)src[i] * scale);
                if (ref_q > out_range) ref_q = out_range;
                if (ref_q < -out_range) ref_q = -out_range;
                if (abs((int)dst[i] - ref_q) > 1) worse++;
            }

            printf("  n=%4d range=%3d: in_range=%d worse=%d %s\n",
                   n, out_range, ok, worse, (ok && worse == 0) ? "OK" : "FAIL");
            if (ok && worse == 0) tests_passed++; else tests_failed++;

            free(src); free(dst);
        }
    }
}

/* ================================================================
 *  Test 3: Performance comparison
 * ================================================================ */

static void test_performance(void) {
    printf("=== Test 3: Performance ===\n");

    int n = 2560;
    int iters = 10000;
    float eps = 1e-5f;

    int8_t *src = (int8_t *)malloc(n);
    int8_t *dst = (int8_t *)malloc(n);
    float *src_f = (float *)malloc(n * sizeof(float));
    float *dst_f = (float *)malloc(n * sizeof(float));
    mtfp21_t *src_m = (mtfp21_t *)malloc(n * sizeof(mtfp21_t));
    mtfp21_t *dst_m = (mtfp21_t *)malloc(n * sizeof(mtfp21_t));
    mtfp21_t eps_m = mtfp21_from_float(eps);

    srand(42);
    for (int i = 0; i < n; i++) {
        src[i] = (int8_t)((rand() % 161) - 80);
        src_f[i] = (float)src[i];
        src_m[i] = mtfp21_from_float(src_f[i]);
    }

    struct timespec t0, t1;

    /* AVX2 integer kernel */
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int it = 0; it < iters; it++) {
        ternary_rmsnorm_avx2(dst, src, n, eps, 80);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double avx2_ns = elapsed_us(t0, t1) * 1000.0 / iters;

    /* Float32 RMSNorm */
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int it = 0; it < iters; it++) {
        float sum = 0.0f;
        for (int i = 0; i < n; i++) sum += src_f[i] * src_f[i];
        float sc = 1.0f / sqrtf(sum / n + eps);
        for (int i = 0; i < n; i++) dst_f[i] = src_f[i] * sc;
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double f32_ns = elapsed_us(t0, t1) * 1000.0 / iters;

    /* MTFP21 RMSNorm */
    int mtfp_iters = 100;  /* fewer iters — it's slow */
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int it = 0; it < mtfp_iters; it++) {
        mtfp21_rmsnorm_pure(dst_m, src_m, n, eps_m);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double mtfp_ns = elapsed_us(t0, t1) * 1000.0 / mtfp_iters;

    printf("  n=%d\n", n);
    printf("  %-20s %10.0f ns/call\n", "AVX2 integer:", avx2_ns);
    printf("  %-20s %10.0f ns/call\n", "float32:", f32_ns);
    printf("  %-20s %10.0f ns/call\n", "MTFP21:", mtfp_ns);
    printf("\n");
    printf("  AVX2 vs float32:   %.1fx %s\n",
           f32_ns > avx2_ns ? f32_ns / avx2_ns : avx2_ns / f32_ns,
           f32_ns > avx2_ns ? "faster" : "slower");
    printf("  AVX2 vs MTFP21:    %.0fx faster\n", mtfp_ns / avx2_ns);
    printf("  float32 vs MTFP21: %.0fx faster\n", mtfp_ns / f32_ns);

    tests_passed++;  /* measurement, not pass/fail */

    free(src); free(dst); free(src_f); free(dst_f); free(src_m); free(dst_m);
}

/* ================================================================
 *  Test 4: Adversarial inputs
 * ================================================================ */

static void test_adversarial(void) {
    printf("=== Test 4: Adversarial inputs ===\n");

    int n = 2560;
    int out_range = 80;
    int8_t *src = (int8_t *)malloc(n);
    int8_t *dst = (int8_t *)malloc(n);

    /* All zeros */
    memset(src, 0, n);
    ternary_rmsnorm_avx2(dst, src, n, 1e-5f, out_range);
    int all_zero = 1;
    for (int i = 0; i < n; i++) if (dst[i] != 0) { all_zero = 0; break; }
    printf("  all_zeros: %s\n", all_zero ? "OK (all zero output)" : "FAIL");
    if (all_zero) tests_passed++; else tests_failed++;

    /* All same value */
    memset(src, 40, n);
    ternary_rmsnorm_avx2(dst, src, n, 1e-5f, out_range);
    int all_same = 1;
    int8_t first = dst[0];
    for (int i = 1; i < n; i++) if (dst[i] != first) { all_same = 0; break; }
    printf("  all_same (40): output=%d, all_equal=%s\n", first, all_same ? "YES" : "NO");
    if (all_same && first > 0) tests_passed++; else tests_failed++;

    /* Alternating max */
    for (int i = 0; i < n; i++) src[i] = (i % 2 == 0) ? 80 : -80;
    ternary_rmsnorm_avx2(dst, src, n, 1e-5f, out_range);
    int alt_ok = 1;
    for (int i = 0; i < n; i++) {
        if (dst[i] > out_range || dst[i] < -out_range) { alt_ok = 0; break; }
    }
    printf("  alternating: in_range=%s\n", alt_ok ? "OK" : "FAIL");
    if (alt_ok) tests_passed++; else tests_failed++;

    /* Single non-zero */
    memset(src, 0, n);
    src[0] = 80;
    ternary_rmsnorm_avx2(dst, src, n, 1e-5f, out_range);
    printf("  single_nonzero: dst[0]=%d (expect large), dst[1]=%d (expect 0)\n", dst[0], dst[1]);
    if (dst[0] != 0 && dst[1] == 0) tests_passed++; else tests_failed++;

    free(src); free(dst);
}

/* ================================================================
 *  Main
 * ================================================================ */

int main(void) {
    printf("Shirley AVX2 Kernel Test Suite\n");
    printf("==============================\n\n");

    test_correctness();
    test_dimensions();
    test_performance();
    test_adversarial();

    printf("\n==============================\n");
    printf("Results: %d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
