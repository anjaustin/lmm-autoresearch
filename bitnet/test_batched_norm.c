/*
 * test_batched_norm.c — Validate batched normalization math
 *
 * Compare: per-row normalization vs batched (find global max, uniform shift)
 * for ternary matmul output.
 *
 * Build: gcc -O2 -mavx2 -o test_batched_norm test_batched_norm.c -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include "shirley_mtfp21.h"

static int tests_passed = 0;
static int tests_failed = 0;

/* ================================================================
 *  Per-row normalization (current approach)
 * ================================================================ */

static void normalize_per_row(
    mtfp21_t * dst,
    const int32_t * raw,     /* raw dot products */
    int n_rows,
    int8_t block_exp,
    float weight_scale
) {
    mtfp21_t ws = mtfp21_from_float(weight_scale);
    for (int i = 0; i < n_rows; i++) {
        mtfp21_t r;
        r.mantissa = raw[i];
        r.exponent = block_exp;
        /* Normalize */
        while (r.mantissa != 0 && llabs(r.mantissa) > MTFP21_MANT_MAX) {
            int32_t rem = r.mantissa % 3;
            r.mantissa /= 3;
            if (rem == 2) r.mantissa++;
            else if (rem == -2) r.mantissa--;
            r.exponent++;
        }
        while (r.mantissa != 0 && llabs(r.mantissa) * 3 <= MTFP21_MANT_MAX
               && r.exponent > -MTFP21_EXP_MAX) {
            r.mantissa *= 3;
            r.exponent--;
        }
        dst[i] = mtfp21_mul(r, ws);
    }
}

/* ================================================================
 *  Batched normalization (proposed approach)
 * ================================================================ */

static void normalize_batched(
    mtfp21_t * dst,
    const int32_t * raw,
    int n_rows,
    int8_t block_exp,
    float weight_scale
) {
    /* Step 1: Find max absolute value across all rows */
    int64_t max_abs = 0;
    for (int i = 0; i < n_rows; i++) {
        int64_t a = raw[i] > 0 ? (int64_t)raw[i] : -(int64_t)raw[i];
        if (a > max_abs) max_abs = a;
    }

    if (max_abs == 0) {
        for (int i = 0; i < n_rows; i++) dst[i] = (mtfp21_t){0, 0};
        return;
    }

    /* Step 2: Compute uniform trit-shift from max */
    int shift_up = 0;   /* shift mantissa UP (multiply by 3) to maximize precision */
    int shift_down = 0;  /* shift mantissa DOWN (divide by 3) to fit MTFP21 */

    if (max_abs > MTFP21_MANT_MAX) {
        /* Need to shift down */
        int64_t test = max_abs;
        while (test > MTFP21_MANT_MAX) {
            test /= 3;
            shift_down++;
        }
    } else {
        /* Can shift up for more precision */
        int64_t test = max_abs;
        while (test * 3 <= MTFP21_MANT_MAX) {
            test *= 3;
            shift_up++;
        }
    }

    int net_exp_adjust = shift_down - shift_up;

    /* Step 3: Apply uniform shift to all rows + multiply by weight_scale */
    mtfp21_t ws = mtfp21_from_float(weight_scale);

    /* Step 3: Apply global shift-down to fit int32, then per-element
     * precision recovery (shift up to maximize each mantissa).
     * The global shift-down is uniform (one computation).
     * The per-element shift-up is cheap (small mantissas shift more). */
    for (int i = 0; i < n_rows; i++) {
        if (raw[i] == 0) { dst[i] = (mtfp21_t){0, 0}; continue; }

        int64_t m = (int64_t)raw[i];
        int exp = (int)block_exp;

        /* Global shift-down: if max_abs > MANT_MAX, all values shift down */
        if (shift_down > 0 && shift_down < 32) {
            int64_t divisor = POW3[shift_down];
            int64_t half = divisor / 2;
            if (m >= 0) m = (m + half) / divisor;
            else        m = -((-m + half) / divisor);
            exp += shift_down;
        }

        /* Per-element precision recovery: shift up to maximize mantissa */
        while (m != 0 && llabs(m) * 3 <= MTFP21_MANT_MAX && exp > -MTFP21_EXP_MAX) {
            m *= 3;
            exp--;
        }

        mtfp21_t r;
        r.mantissa = (int32_t)m;
        r.exponent = (int8_t)exp;
        dst[i] = mtfp21_mul(r, ws);
    }
}

/* ================================================================
 *  Tests
 * ================================================================ */

static void test_small(void) {
    printf("--- test_small ---\n");
    int32_t raw[] = {100, -200, 50, 0, 300, -150, 75, -25};
    int n = 8;
    int8_t block_exp = -10;
    float wscale = 1.5f;

    mtfp21_t per_row[8], batched[8];
    normalize_per_row(per_row, raw, n, block_exp, wscale);
    normalize_batched(batched, raw, n, block_exp, wscale);

    int pass = 0;
    float max_err = 0;
    for (int i = 0; i < n; i++) {
        float a = mtfp21_to_float(per_row[i]);
        float b = mtfp21_to_float(batched[i]);
        float err = (fabsf(a) > 1e-10f) ? fabsf((b - a) / a) : fabsf(b - a);
        if (err > max_err) max_err = err;
        if (err < 1e-5f) pass++;
    }
    printf("  %d/%d match (max err: %.2e)\n", pass, n, max_err);
    if (pass == n) tests_passed++; else tests_failed++;
}

static void test_realistic(void) {
    printf("--- test_realistic (n=2560, typical matmul output) ---\n");

    int n = 2560;
    int32_t * raw = (int32_t *)malloc(n * sizeof(int32_t));

    /* Simulate actual ternary matmul output:
     * dot product of 2560 int16 activations (post-norm, ±1000) × ternary weights */
    srand(42);
    {
        int16_t act[2560];
        for (int j = 0; j < 2560; j++) act[j] = (int16_t)((rand() % 2000) - 1000);
        for (int i = 0; i < n; i++) {
            int32_t dot = 0;
            for (int j = 0; j < 2560; j++) {
                int w = (rand() % 3) - 1;
                dot += (int32_t)act[j] * w;
            }
            raw[i] = dot;
        }
    }

    int8_t block_exp = -12;
    float wscale = 2.1f;

    mtfp21_t * per_row = (mtfp21_t *)malloc(n * sizeof(mtfp21_t));
    mtfp21_t * batched = (mtfp21_t *)malloc(n * sizeof(mtfp21_t));

    normalize_per_row(per_row, raw, n, block_exp, wscale);
    normalize_batched(batched, raw, n, block_exp, wscale);

    int pass = 0;
    float max_err = 0;
    double sum_err = 0;
    for (int i = 0; i < n; i++) {
        float a = mtfp21_to_float(per_row[i]);
        float b = mtfp21_to_float(batched[i]);
        float err = (fabsf(a) > 1e-10f) ? fabsf((b - a) / a) : fabsf(b - a);
        if (err > max_err) max_err = err;
        sum_err += err;
        if (err < 1e-4f) pass++;
    }
    printf("  %d/%d match (avg err: %.2e, max err: %.2e)\n",
           pass, n, sum_err / n, max_err);
    if (pass == n) tests_passed++; else tests_failed++;

    free(raw); free(per_row); free(batched);
}

static void test_statistical(void) {
    printf("--- test_statistical (1000 seeds, n=2560) ---\n");

    int n = 2560;
    int32_t * raw = (int32_t *)malloc(n * sizeof(int32_t));
    mtfp21_t * per_row = (mtfp21_t *)malloc(n * sizeof(mtfp21_t));
    mtfp21_t * batched = (mtfp21_t *)malloc(n * sizeof(mtfp21_t));

    int trials = 1000;
    float global_max_err = 0;
    double global_sum_err = 0;
    int total_mismatch = 0;

    for (int t = 0; t < trials; t++) {
        srand(t);
        {
            int16_t act[2560];
            for (int j = 0; j < 2560; j++) act[j] = (int16_t)((rand() % 2000) - 1000);
            for (int i = 0; i < n; i++) {
                int32_t dot = 0;
                for (int j = 0; j < 2560; j++) {
                    int w = (rand() % 3) - 1;
                    dot += (int32_t)act[j] * w;
                }
                raw[i] = dot;
            }
        }

        int8_t block_exp = (int8_t)(-8 - (t % 10));
        float wscale = 0.5f + (float)(t % 20) * 0.2f;

        normalize_per_row(per_row, raw, n, block_exp, wscale);
        normalize_batched(batched, raw, n, block_exp, wscale);

        for (int i = 0; i < n; i++) {
            float a = mtfp21_to_float(per_row[i]);
            float b = mtfp21_to_float(batched[i]);
            float err = (fabsf(a) > 1e-10f) ? fabsf((b - a) / a) : fabsf(b - a);
            if (err > global_max_err) global_max_err = err;
            global_sum_err += err;
            if (err > 1e-4f) total_mismatch++;
        }
    }

    double avg_err = global_sum_err / (trials * n);
    printf("  Trials: %d, Elements per trial: %d\n", trials, n);
    printf("  Avg error: %.2e\n", avg_err);
    printf("  Max error: %.2e\n", global_max_err);
    printf("  Mismatches (>1e-4): %d / %d (%.4f%%)\n",
           total_mismatch, trials * n, 100.0 * total_mismatch / (trials * n));

    if (global_max_err < 1e-3f) {
        printf("  PASS (max err < 1e-3)\n");
        tests_passed++;
    } else {
        printf("  FAIL\n");
        tests_failed++;
    }

    free(raw); free(per_row); free(batched);
}

static void test_edge_cases(void) {
    printf("--- test_edge_cases ---\n");

    /* All zeros */
    {
        int32_t raw[] = {0, 0, 0, 0};
        mtfp21_t a[4], b[4];
        normalize_per_row(a, raw, 4, -10, 1.0f);
        normalize_batched(b, raw, 4, -10, 1.0f);
        int ok = 1;
        for (int i = 0; i < 4; i++) {
            if (a[i].mantissa != b[i].mantissa) ok = 0;
        }
        printf("  All zeros: %s\n", ok ? "PASS" : "FAIL");
        if (ok) tests_passed++; else tests_failed++;
    }

    /* Single large value */
    {
        int32_t raw[] = {100000000, 1, -1, 0};
        mtfp21_t a[4], b[4];
        normalize_per_row(a, raw, 4, -15, 2.0f);
        normalize_batched(b, raw, 4, -15, 2.0f);
        float max_err = 0;
        for (int i = 0; i < 4; i++) {
            float fa = mtfp21_to_float(a[i]);
            float fb = mtfp21_to_float(b[i]);
            float err = (fabsf(fa) > 1e-10f) ? fabsf((fb - fa) / fa) : fabsf(fb);
            if (err > max_err) max_err = err;
        }
        printf("  Large outlier: max_err=%.2e %s\n", max_err, max_err < 0.01f ? "PASS" : "FAIL");
        if (max_err < 0.01f) tests_passed++; else tests_failed++;
    }

    /* All same value */
    {
        int32_t raw[256];
        for (int i = 0; i < 256; i++) raw[i] = 12345;
        mtfp21_t a[256], b[256];
        normalize_per_row(a, raw, 256, -10, 1.5f);
        normalize_batched(b, raw, 256, -10, 1.5f);
        float max_err = 0;
        for (int i = 0; i < 256; i++) {
            float fa = mtfp21_to_float(a[i]);
            float fb = mtfp21_to_float(b[i]);
            float err = fabsf(fa) > 1e-10f ? fabsf((fb - fa) / fa) : 0;
            if (err > max_err) max_err = err;
        }
        printf("  All same: max_err=%.2e %s\n", max_err, max_err < 1e-6f ? "PASS" : "FAIL");
        if (max_err < 1e-6f) tests_passed++; else tests_failed++;
    }
}

int main(void) {
    printf("Batched Normalization Validation\n");
    printf("================================\n\n");

    test_small();
    test_realistic();
    test_statistical();
    test_edge_cases();

    printf("\n================================\n");
    printf("Results: %d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
