/*
 * profile_mtfp21.c — Profile each atomic operation in the MTFP21 RMSNorm pipeline
 *
 * Measures: conversion, multiply, add, div_scalar, rsqrt, and the full pipeline
 * broken down by phase.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include "shirley_mtfp21.h"

#define N 2560
#define ITERS 1000

static double elapsed_us(struct timespec t0, struct timespec t1) {
    return (t1.tv_sec - t0.tv_sec) * 1e6 + (t1.tv_nsec - t0.tv_nsec) / 1e3;
}

static float randn(void) {
    float u1 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 1.0f);
    float u2 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 1.0f);
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2);
}

int main(void) {
    printf("MTFP21 RMSNorm Profiler — n=%d, %d iterations\n", N, ITERS);
    printf("=========================================\n\n");

    /* Prepare data */
    srand(42);
    float src_f[N];
    mtfp21_t src_m[N], dst_m[N];
    for (int i = 0; i < N; i++) {
        src_f[i] = randn();
        src_m[i] = mtfp21_from_float(src_f[i]);
    }
    mtfp21_t eps_m = mtfp21_from_float(1e-5f);

    struct timespec t0, t1;
    double us;
    volatile mtfp21_t sink;  /* prevent optimizer from eliding */
    volatile float sink_f;

    /* ---- Atomic: from_float ---- */
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int it = 0; it < ITERS; it++) {
        for (int i = 0; i < N; i++) {
            sink = mtfp21_from_float(src_f[i]);
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    us = elapsed_us(t0, t1) / ITERS;
    printf("  %-25s %8.1f us/call  (%5.1f ns/element)\n", "from_float (N)", us, us*1000/N);

    /* ---- Atomic: to_float ---- */
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int it = 0; it < ITERS; it++) {
        for (int i = 0; i < N; i++) {
            sink_f = mtfp21_to_float(src_m[i]);
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    us = elapsed_us(t0, t1) / ITERS;
    printf("  %-25s %8.1f us/call  (%5.1f ns/element)\n", "to_float (N)", us, us*1000/N);

    /* ---- Atomic: multiply ---- */
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int it = 0; it < ITERS; it++) {
        for (int i = 0; i < N; i++) {
            sink = mtfp21_mul(src_m[i], src_m[i]);
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    us = elapsed_us(t0, t1) / ITERS;
    printf("  %-25s %8.1f us/call  (%5.1f ns/element)\n", "mul (N squares)", us, us*1000/N);

    /* ---- Atomic: add (accumulation) ---- */
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int it = 0; it < ITERS; it++) {
        mtfp21_t acc = {0, 0};
        for (int i = 0; i < N; i++) {
            acc = mtfp21_add(acc, src_m[i]);
        }
        sink = acc;
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    us = elapsed_us(t0, t1) / ITERS;
    printf("  %-25s %8.1f us/call  (%5.1f ns/element)\n", "add (N accumulate)", us, us*1000/N);

    /* ---- Atomic: div_scalar ---- */
    mtfp21_t test_sum = mtfp21_from_float(2584.0f);
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int it = 0; it < ITERS * 100; it++) {
        sink = mtfp21_div_scalar(test_sum, N);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    us = elapsed_us(t0, t1) / (ITERS * 100);
    printf("  %-25s %8.3f us/call  (1 call per RMSNorm)\n", "div_scalar", us);

    /* ---- Atomic: rsqrt ---- */
    mtfp21_t test_mean = mtfp21_from_float(1.009f);
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int it = 0; it < ITERS * 100; it++) {
        sink = mtfp21_rsqrt(test_mean);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    us = elapsed_us(t0, t1) / (ITERS * 100);
    printf("  %-25s %8.3f us/call  (1 call per RMSNorm)\n", "rsqrt (LUT+2NR)", us);

    /* ---- Atomic: scale multiply (N elements × 1 scalar) ---- */
    mtfp21_t test_scale = mtfp21_from_float(0.995f);
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int it = 0; it < ITERS; it++) {
        for (int i = 0; i < N; i++) {
            dst_m[i] = mtfp21_mul(src_m[i], test_scale);
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    us = elapsed_us(t0, t1) / ITERS;
    printf("  %-25s %8.1f us/call  (%5.1f ns/element)\n", "mul (N × scalar)", us, us*1000/N);

    /* ---- Full pipeline: phase breakdown ---- */
    printf("\n  --- Full RMSNorm pipeline breakdown ---\n");

    double phase_us[5];

    /* Phase 1: Square all elements */
    mtfp21_t squares[N];
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int it = 0; it < ITERS; it++) {
        for (int i = 0; i < N; i++) {
            squares[i] = mtfp21_mul(src_m[i], src_m[i]);
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    phase_us[0] = elapsed_us(t0, t1) / ITERS;

    /* Phase 2: Accumulate sum of squares */
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int it = 0; it < ITERS; it++) {
        mtfp21_t acc = {0, 0};
        for (int i = 0; i < N; i++) {
            acc = mtfp21_add(acc, squares[i]);
        }
        sink = acc;
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    phase_us[1] = elapsed_us(t0, t1) / ITERS;

    /* Phase 3: div + eps + rsqrt */
    mtfp21_t sum_sq = {0, 0};
    for (int i = 0; i < N; i++) {
        sum_sq = mtfp21_add(sum_sq, mtfp21_mul(src_m[i], src_m[i]));
    }
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int it = 0; it < ITERS; it++) {
        mtfp21_t mean = mtfp21_div_scalar(sum_sq, N);
        mtfp21_t mean_eps = mtfp21_add(mean, eps_m);
        sink = mtfp21_rsqrt(mean_eps);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    phase_us[2] = elapsed_us(t0, t1) / ITERS;

    /* Phase 4: Scale all elements */
    mtfp21_t scale = mtfp21_rsqrt(mtfp21_add(mtfp21_div_scalar(sum_sq, N), eps_m));
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int it = 0; it < ITERS; it++) {
        for (int i = 0; i < N; i++) {
            dst_m[i] = mtfp21_mul(src_m[i], scale);
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    phase_us[3] = elapsed_us(t0, t1) / ITERS;

    /* Full pipeline */
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int it = 0; it < ITERS; it++) {
        mtfp21_rmsnorm_pure(dst_m, src_m, N, eps_m);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    phase_us[4] = elapsed_us(t0, t1) / ITERS;

    double total_phases = phase_us[0] + phase_us[1] + phase_us[2] + phase_us[3];
    printf("  %-25s %8.1f us  (%5.1f%%)\n", "1. Square (N mul)", phase_us[0], 100*phase_us[0]/phase_us[4]);
    printf("  %-25s %8.1f us  (%5.1f%%)\n", "2. Accumulate (N add)", phase_us[1], 100*phase_us[1]/phase_us[4]);
    printf("  %-25s %8.1f us  (%5.1f%%)\n", "3. div+eps+rsqrt (1)", phase_us[2], 100*phase_us[2]/phase_us[4]);
    printf("  %-25s %8.1f us  (%5.1f%%)\n", "4. Scale (N mul)", phase_us[3], 100*phase_us[3]/phase_us[4]);
    printf("  %-25s %8.1f us\n", "Sum of phases:", total_phases);
    printf("  %-25s %8.1f us\n", "Full pipeline:", phase_us[4]);

    /* Float32 reference */
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int it = 0; it < ITERS; it++) {
        float sum = 0.0f;
        for (int i = 0; i < N; i++) sum += src_f[i] * src_f[i];
        float sc = 1.0f / sqrtf(sum / N + 1e-5f);
        for (int i = 0; i < N; i++) sink_f = src_f[i] * sc;
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double f32_us = elapsed_us(t0, t1) / ITERS;
    printf("\n  %-25s %8.1f us\n", "float32 RMSNorm:", f32_us);
    printf("  %-25s %8.1fx\n", "MTFP21/float32 ratio:", phase_us[4] / f32_us);

    /* What fraction of time is per-element work vs scalar work? */
    double per_elem = phase_us[0] + phase_us[1] + phase_us[3];
    double scalar = phase_us[2];
    printf("\n  Per-element work (mul+add+scale): %.1f us (%.1f%%)\n", per_elem, 100*per_elem/phase_us[4]);
    printf("  Scalar work (div+rsqrt):          %.1f us (%.1f%%)\n", scalar, 100*scalar/phase_us[4]);
    printf("  -> SIMD vectorization targets the %.1f%% per-element portion\n", 100*per_elem/phase_us[4]);

    return 0;
}
