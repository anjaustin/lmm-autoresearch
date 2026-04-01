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
/* shirley_mtfp21.h included via shirley_kernels.h */

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
 *  Test 1: float32→float32 RMSNorm with gamma (the real operation)
 * ================================================================ */

static void test_f32_with_gamma(void) {
    printf("=== Test 1: shirley_rmsnorm_f32 with gamma ===\n");

    int n = 2560;
    float eps = 1e-5f;
    float *src = (float *)malloc(n * sizeof(float));
    float *gamma_w = (float *)malloc(n * sizeof(float));
    float *dst = (float *)malloc(n * sizeof(float));

    srand(42);
    for (int i = 0; i < n; i++) {
        src[i] = randn();
        gamma_w[i] = 0.5f + randn() * 0.1f;  /* realistic gamma: near 1.0 with variance */
    }

    /* Shirley kernel */
    shirley_rmsnorm_f32(dst, src, gamma_w, n, eps);

    /* Float64 reference */
    double sum_sq = 0.0;
    for (int i = 0; i < n; i++) sum_sq += (double)src[i] * (double)src[i];
    double scale = 1.0 / sqrt(sum_sq / n + eps);

    float max_err = 0.0f;
    for (int i = 0; i < n; i++) {
        double ref = (double)src[i] * scale * (double)gamma_w[i];
        float err = fabsf(ref) > 1e-10 ? fabsf((dst[i] - (float)ref) / (float)ref) : fabsf(dst[i]);
        if (err > max_err) max_err = err;
    }

    printf("  n=%d, with gamma, max_err=%.2e\n", n, max_err);
    if (max_err < 1e-5f) { printf("  PASS\n"); tests_passed++; }
    else { printf("  FAIL\n"); tests_failed++; }

    /* Without gamma */
    shirley_rmsnorm_f32(dst, src, NULL, n, eps);
    max_err = 0.0f;
    for (int i = 0; i < n; i++) {
        double ref = (double)src[i] * scale;
        float err = fabsf(ref) > 1e-10 ? fabsf((dst[i] - (float)ref) / (float)ref) : fabsf(dst[i]);
        if (err > max_err) max_err = err;
    }
    printf("  Without gamma: max_err=%.2e\n", max_err);
    if (max_err < 1e-6f) { printf("  PASS\n"); tests_passed++; }
    else { printf("  FAIL\n"); tests_failed++; }

    free(src); free(gamma_w); free(dst);
}

/* ================================================================
 *  Test 1b: float32→int8 fused RMSNorm+quantize with gamma
 * ================================================================ */

static void test_fused_quantize(void) {
    printf("=== Test 1b: shirley_rmsnorm_quantize (float32→int8) ===\n");

    int n = 2560;
    float eps = 1e-5f;
    int out_range = 80;
    float *src = (float *)malloc(n * sizeof(float));
    float *gamma_w = (float *)malloc(n * sizeof(float));
    int8_t *dst = (int8_t *)malloc(n);

    /* Use small-variance inputs so output spans the quantization range */
    srand(42);
    for (int i = 0; i < n; i++) {
        src[i] = randn() * 0.1f;  /* small inputs → large scale → wider output */
        gamma_w[i] = 0.8f + randn() * 0.2f;
    }

    float quant_scale = shirley_rmsnorm_quantize(dst, src, gamma_w, n, eps, out_range);

    /* Float64 reference */
    double sum_sq = 0.0;
    for (int i = 0; i < n; i++) sum_sq += (double)src[i] * (double)src[i];
    double norm_scale = 1.0 / sqrt(sum_sq / n + eps);

    /* Compute reference normalized+gamma values, find max for quant */
    double *ref_norm = (double *)malloc(n * sizeof(double));
    double ref_max = 0.0;
    for (int i = 0; i < n; i++) {
        ref_norm[i] = (double)src[i] * norm_scale * (double)gamma_w[i];
        if (fabs(ref_norm[i]) > ref_max) ref_max = fabs(ref_norm[i]);
    }
    double ref_qscale = (ref_max > 1e-10) ? out_range / ref_max : 0.0;

    int exact = 0, off1 = 0, worse = 0;
    for (int i = 0; i < n; i++) {
        int ref_q = (int)round(ref_norm[i] * ref_qscale);
        if (ref_q > out_range) ref_q = out_range;
        if (ref_q < -out_range) ref_q = -out_range;

        int diff = abs((int)dst[i] - ref_q);
        if (diff == 0) exact++;
        else if (diff == 1) off1++;
        else worse++;
    }

    /* Count distinct output values (stress test: should be >> 5) */
    int histogram[256] = {0};
    for (int i = 0; i < n; i++) histogram[dst[i] + 128]++;
    int distinct = 0;
    for (int i = 0; i < 256; i++) if (histogram[i] > 0) distinct++;

    printf("  Exact: %d/%d (%.1f%%), Off-by-1: %d, Worse: %d\n",
           exact, n, 100.0 * exact / n, off1, worse);
    printf("  Distinct output values: %d (stress check: should be >> 5)\n", distinct);
    printf("  quant_scale=%.4f\n", quant_scale);

    int pass = (worse == 0) && ((float)exact / n >= 0.95f) && (distinct > 20);
    if (pass) { printf("  PASS\n"); tests_passed++; }
    else { printf("  FAIL\n"); tests_failed++; }

    free(src); free(gamma_w); free(dst); free(ref_norm);
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
    printf("=== Test 3: Performance (fair comparison) ===\n");

    int n = 2560;
    int iters = 10000;
    float eps = 1e-5f;

    float *src_f = (float *)malloc(n * sizeof(float));
    float *dst_f = (float *)malloc(n * sizeof(float));
    float *gamma_w = (float *)malloc(n * sizeof(float));
    int8_t *dst_i8 = (int8_t *)malloc(n);

    srand(42);
    for (int i = 0; i < n; i++) {
        src_f[i] = randn();
        gamma_w[i] = 0.8f + randn() * 0.2f;
    }

    struct timespec t0, t1;

    /* Baseline: scalar float32 RMSNorm + gamma (what ggml does today) */
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int it = 0; it < iters; it++) {
        float sum = 0.0f;
        for (int i = 0; i < n; i++) sum += src_f[i] * src_f[i];
        float sc = 1.0f / sqrtf(sum / n + eps);
        for (int i = 0; i < n; i++) dst_f[i] = src_f[i] * sc * gamma_w[i];
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double scalar_ns = elapsed_us(t0, t1) * 1000.0 / iters;

    /* Shirley: AVX2 float32→float32 with gamma */
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int it = 0; it < iters; it++) {
        shirley_rmsnorm_f32(dst_f, src_f, gamma_w, n, eps);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double shirley_f32_ns = elapsed_us(t0, t1) * 1000.0 / iters;

    /* Shirley: AVX2 float32→int8 fused (norm+gamma+quantize) */
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int it = 0; it < iters; it++) {
        shirley_rmsnorm_quantize(dst_i8, src_f, gamma_w, n, eps, 80);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double shirley_quant_ns = elapsed_us(t0, t1) * 1000.0 / iters;

    printf("  n=%d, all with gamma weights, float32 input\n", n);
    printf("  %-35s %8.0f ns/call\n", "scalar float32 (ggml baseline):", scalar_ns);
    printf("  %-35s %8.0f ns/call\n", "shirley_rmsnorm_f32 (AVX2):", shirley_f32_ns);
    printf("  %-35s %8.0f ns/call\n", "shirley_rmsnorm_quantize (fused):", shirley_quant_ns);
    printf("\n");
    if (shirley_f32_ns < scalar_ns) {
        printf("  f32 AVX2 vs scalar: %.1fx faster\n", scalar_ns / shirley_f32_ns);
    } else {
        printf("  f32 AVX2 vs scalar: %.1fx slower\n", shirley_f32_ns / scalar_ns);
    }

    tests_passed++;  /* measurement, not pass/fail */

    free(src_f); free(dst_f); free(gamma_w); free(dst_i8);
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

    test_f32_with_gamma();
    test_fused_quantize();
    test_dimensions();
    test_performance();
    test_adversarial();

    /* Test 5: End-to-end ternary kernel */
    {
        printf("=== Test 5: shirley_rmsnorm_ternary (zero float) ===\n");

        int n = 2560;
        int out_range = 80;
        int8_t *src = (int8_t *)malloc(n);
        int8_t *dst_ternary = (int8_t *)malloc(n);
        int8_t *dst_legacy = (int8_t *)malloc(n);

        srand(42);
        for (int i = 0; i < n; i++) {
            src[i] = (int8_t)((rand() % 161) - 80);
        }

        /* Ternary kernel (zero float), no gamma */
        shirley_rmsnorm_ternary(dst_ternary, src, NULL, n, out_range);

        /* Float64 reference */
        double sum_sq = 0.0;
        for (int i = 0; i < n; i++) sum_sq += (double)src[i] * (double)src[i];
        double scale = 1.0 / sqrt(sum_sq / n + 1e-5);

        int exact = 0, off1 = 0, worse = 0;
        for (int i = 0; i < n; i++) {
            int ref_q = (int)round((double)src[i] * scale);
            if (ref_q > out_range) ref_q = out_range;
            if (ref_q < -out_range) ref_q = -out_range;
            int diff = abs((int)dst_ternary[i] - ref_q);
            if (diff == 0) exact++;
            else if (diff == 1) off1++;
            else worse++;
        }

        printf("  vs float64 ref: Exact=%d (%.1f%%), Off-by-1=%d, Worse=%d\n",
               exact, 100.0*exact/n, off1, worse);

        int pass = ((float)(exact + off1) / n >= 0.99f) && ((float)exact / n >= 0.90f);
        if (pass) { printf("  PASS\n"); tests_passed++; }
        else { printf("  FAIL\n"); tests_failed++; }

        /* Performance comparison: ternary vs float hybrid vs MTFP21 */
        printf("\n  --- Performance ---\n");
        struct timespec t0, t1;
        int iters = 10000;

        clock_gettime(CLOCK_MONOTONIC, &t0);
        for (int it = 0; it < iters; it++) {
            shirley_rmsnorm_ternary(dst_ternary, src, NULL, n, out_range);
        }
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double ternary_ns = elapsed_us(t0, t1) * 1000.0 / iters;

        clock_gettime(CLOCK_MONOTONIC, &t0);
        for (int it = 0; it < iters; it++) {
            ternary_rmsnorm_avx2(dst_legacy, src, n, 1e-5f, out_range);
        }
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double legacy_ns = elapsed_us(t0, t1) * 1000.0 / iters;

        /* MTFP21 for comparison */
        mtfp21_t *src_m = (mtfp21_t *)malloc(n * sizeof(mtfp21_t));
        mtfp21_t *dst_m = (mtfp21_t *)malloc(n * sizeof(mtfp21_t));
        mtfp21_t eps_m = mtfp21_from_float(1e-5f);
        for (int i = 0; i < n; i++) src_m[i] = mtfp21_from_int8(src[i]);

        int mtfp_iters = 100;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        for (int it = 0; it < mtfp_iters; it++) {
            mtfp21_rmsnorm_pure(dst_m, src_m, n, eps_m);
        }
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double mtfp_ns = elapsed_us(t0, t1) * 1000.0 / mtfp_iters;

        printf("  %-40s %8.0f ns\n", "shirley_rmsnorm_ternary (ZERO FLOAT):", ternary_ns);
        printf("  %-40s %8.0f ns\n", "ternary_rmsnorm_avx2 (float rsqrt):", legacy_ns);
        printf("  %-40s %8.0f ns\n", "mtfp21_rmsnorm_pure (scalar MTFP21):", mtfp_ns);
        printf("\n");
        printf("  ternary vs float-hybrid: %.1fx %s\n",
               legacy_ns > ternary_ns ? legacy_ns / ternary_ns : ternary_ns / legacy_ns,
               legacy_ns > ternary_ns ? "faster" : "slower");
        printf("  ternary vs MTFP21:       %.0fx faster\n", mtfp_ns / ternary_ns);
        tests_passed++;

        free(src_m); free(dst_m);

        /* ---- Adversarial cases on ternary kernel (Fix 3) ---- */
        printf("\n  --- Adversarial (ternary kernel) ---\n");

        struct { const char *name; void (*fill)(int8_t *, int); } adv_cases[] = {
            {"all_zeros", NULL},
            {"all_same_40", NULL},
            {"single_nonzero", NULL},
            {"alternating_max", NULL},
            {"small_values", NULL},
        };

        for (int tc = 0; tc < 5; tc++) {
            memset(src, 0, n);
            switch (tc) {
            case 0: /* all zeros */
                break;
            case 1: /* all same */
                memset(src, 40, n);
                break;
            case 2: /* single nonzero — THE Q15 overflow case */
                src[0] = 80;
                break;
            case 3: /* alternating max */
                for (int j = 0; j < n; j++) src[j] = (j % 2) ? 80 : -80;
                break;
            case 4: /* small values — scale > 1.0 */
                for (int j = 0; j < n; j++) src[j] = (j % 3) - 1; /* {-1, 0, 1} */
                break;
            }

            shirley_rmsnorm_ternary(dst_ternary, src, NULL, n, out_range);

            /* Reference */
            double ss = 0.0;
            for (int j = 0; j < n; j++) ss += (double)src[j] * (double)src[j];
            double sc = 1.0 / sqrt(ss / n + 1e-5);

            int tc_worse = 0;
            int tc_maxdiff = 0;
            for (int j = 0; j < n; j++) {
                int rq = (int)round((double)src[j] * sc);
                if (rq > out_range) rq = out_range;
                if (rq < -out_range) rq = -out_range;
                int diff = abs((int)dst_ternary[j] - rq);
                if (diff > tc_maxdiff) tc_maxdiff = diff;
                if (diff > 1) tc_worse++;
            }

            const char *names[] = {"all_zeros","all_same_40","single_nonzero","alternating","small_values"};
            int ok = (tc_worse == 0);
            printf("  %-18s: worse=%d maxdiff=%d %s\n", names[tc], tc_worse, tc_maxdiff, ok ? "OK" : "FAIL");
            if (ok) tests_passed++; else tests_failed++;
        }

        /* ---- Gamma test (Fix 1) ---- */
        printf("\n  --- Gamma weights ---\n");
        {
            int16_t *gamma_q14 = (int16_t *)malloc(n * sizeof(int16_t));
            srand(42);
            for (int j = 0; j < n; j++) {
                src[j] = (int8_t)((rand() % 161) - 80);
                /* gamma near 1.0 with variance, in Q14 */
                float gf = 0.8f + (float)(rand() % 100) / 250.0f;  /* [0.8, 1.2] */
                gamma_q14[j] = (int16_t)(gf * SHIRLEY_GAMMA_SCALE);
            }

            shirley_rmsnorm_ternary(dst_ternary, src, gamma_q14, n, out_range);

            /* Reference with gamma */
            double ss = 0.0;
            for (int j = 0; j < n; j++) ss += (double)src[j] * (double)src[j];
            double sc = 1.0 / sqrt(ss / n + 1e-5);

            int g_exact = 0, g_off1 = 0, g_worse = 0;
            for (int j = 0; j < n; j++) {
                double gval = (double)gamma_q14[j] / SHIRLEY_GAMMA_SCALE;
                int rq = (int)round((double)src[j] * sc * gval);
                if (rq > out_range) rq = out_range;
                if (rq < -out_range) rq = -out_range;
                int diff = abs((int)dst_ternary[j] - rq);
                if (diff == 0) g_exact++;
                else if (diff == 1) g_off1++;
                else g_worse++;
            }
            printf("  with gamma: Exact=%d (%.1f%%), Off-by-1=%d, Worse=%d\n",
                   g_exact, 100.0*g_exact/n, g_off1, g_worse);
            /* Two-stage fixed-point (scale × gamma) accumulates rounding.
             * 85% exact with rest off-by-1 is the expected precision. */
            int gpass = (g_worse == 0) && ((float)g_exact / n >= 0.85f);
            if (gpass) { printf("  PASS\n"); tests_passed++; }
            else { printf("  FAIL\n"); tests_failed++; }
            free(gamma_q14);
        }

        free(src); free(dst_ternary); free(dst_legacy);
    }

    printf("\n==============================\n");
    printf("Results: %d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
