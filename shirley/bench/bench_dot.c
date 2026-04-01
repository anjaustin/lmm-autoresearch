/*
 * bench_dot.c — Ternary vs Float32 dot product benchmark
 *
 * Phase 1, Step 1.1 of Shirley validation.
 * Measures sign_epi8 ternary dot product against mulps float32 dot product
 * on the actual hardware (Ryzen 5 PRO 5675U).
 *
 * Build:  gcc -O3 -mavx2 -march=native -o bench_dot bench_dot.c -lm
 * Run:    ./bench_dot
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <immintrin.h>

/* ================================================================
 *  Configuration
 * ================================================================ */

#define WARMUP_ITERS   1000000
#define BENCH_ITERS   10000000
#define NUM_SIZES      7

static const int SIZES[NUM_SIZES] = { 16, 32, 64, 128, 256, 512, 1024 };

/* ================================================================
 *  Cycle counter (rdtscp)
 * ================================================================ */

static inline uint64_t rdtscp(void) {
    unsigned lo, hi;
    __asm__ volatile("rdtscp" : "=a"(lo), "=d"(hi) : : "ecx");
    return ((uint64_t)hi << 32) | lo;
}

/* ================================================================
 *  Ternary dot product (sign_epi8 + add_epi8)
 *  Extracted from sstt_v2.c:272-289
 * ================================================================ */

static inline int32_t ternary_dot(const int8_t *a, const int8_t *b, int n) {
    __m256i acc = _mm256_setzero_si256();
    int i;
    for (i = 0; i + 32 <= n; i += 32) {
        __m256i va = _mm256_loadu_si256((const __m256i *)(a + i));
        __m256i vb = _mm256_loadu_si256((const __m256i *)(b + i));
        __m256i prod = _mm256_sign_epi8(va, vb);
        acc = _mm256_add_epi8(acc, prod);
    }
    /* widen epi8 -> epi16 -> epi32 for horizontal sum */
    __m256i lo16 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(acc));
    __m256i hi16 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(acc, 1));
    __m256i sum16 = _mm256_add_epi16(lo16, hi16);
    __m256i sum32 = _mm256_madd_epi16(sum16, _mm256_set1_epi16(1));
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(sum32),
                              _mm256_extracti128_si256(sum32, 1));
    s = _mm_hadd_epi32(s, s);
    s = _mm_hadd_epi32(s, s);
    int32_t result = _mm_cvtsi128_si32(s);

    /* handle tail (< 32 remaining elements) */
    for (; i < n; i++) {
        int8_t av = a[i], bv = b[i];
        result += (bv > 0) ? av : (bv < 0) ? -av : 0;
    }
    return result;
}

/* ================================================================
 *  Float32 dot product (mulps + addps)
 * ================================================================ */

static inline float float32_dot(const float *a, const float *b, int n) {
    __m256 acc = _mm256_setzero_ps();
    int i;
    for (i = 0; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        acc = _mm256_add_ps(acc, _mm256_mul_ps(va, vb));
    }
    /* horizontal sum */
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 sum4 = _mm_add_ps(lo, hi);
    sum4 = _mm_hadd_ps(sum4, sum4);
    sum4 = _mm_hadd_ps(sum4, sum4);
    float result;
    _mm_store_ss(&result, sum4);

    /* handle tail */
    for (; i < n; i++) {
        result += a[i] * b[i];
    }
    return result;
}

/* ================================================================
 *  Data generation
 * ================================================================ */

static void fill_ternary(int8_t *buf, int n) {
    for (int i = 0; i < n; i++) {
        int r = rand() % 3;
        buf[i] = (int8_t)(r - 1);  /* {-1, 0, +1} */
    }
}

static void fill_float_from_ternary(float *dst, const int8_t *src, int n) {
    for (int i = 0; i < n; i++) {
        dst[i] = (float)src[i];
    }
}

/* ================================================================
 *  Benchmark helpers
 * ================================================================ */

static double get_time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}

/* ================================================================
 *  Main
 * ================================================================ */

int main(void) {
    srand(42);

    printf("Shirley Phase 1 — Ternary vs Float32 Dot Product Benchmark\n");
    printf("CPU: Ryzen 5 PRO 5675U (AVX2)\n");
    printf("Warmup: %d iters, Benchmark: %d iters per size\n\n", WARMUP_ITERS, BENCH_ITERS);

    /* allocate max size with alignment */
    int max_n = SIZES[NUM_SIZES - 1];
    int8_t *ta = (int8_t *)aligned_alloc(32, max_n);
    int8_t *tb = (int8_t *)aligned_alloc(32, max_n);
    float  *fa = (float *)aligned_alloc(32, max_n * sizeof(float));
    float  *fb = (float *)aligned_alloc(32, max_n * sizeof(float));

    if (!ta || !tb || !fa || !fb) {
        fprintf(stderr, "Allocation failed\n");
        return 1;
    }

    /* fill with ternary values */
    fill_ternary(ta, max_n);
    fill_ternary(tb, max_n);
    fill_float_from_ternary(fa, ta, max_n);
    fill_float_from_ternary(fb, tb, max_n);

    printf("%-8s | %-14s %-14s %-10s | %-14s %-14s %-10s | %-8s | %s\n",
           "Size", "Tern ns/dot", "Tern cyc/dot", "Tern Mdot/s",
           "FP32 ns/dot", "FP32 cyc/dot", "FP32 Mdot/s",
           "Speedup", "Match?");
    printf("---------+-------------------------------------------+"
           "-------------------------------------------+----------+--------\n");

    for (int si = 0; si < NUM_SIZES; si++) {
        int n = SIZES[si];

        /* verify correctness: ternary dot should equal float32 dot */
        int32_t tern_result = ternary_dot(ta, tb, n);
        float   fp32_result = float32_dot(fa, fb, n);
        int match = (tern_result == (int32_t)fp32_result);

        /* warmup */
        volatile int32_t sink_i = 0;
        volatile float sink_f = 0;
        for (int i = 0; i < WARMUP_ITERS; i++) {
            sink_i += ternary_dot(ta, tb, n);
            sink_f += float32_dot(fa, fb, n);
        }
        (void)sink_i;
        (void)sink_f;

        /* benchmark ternary */
        double t0 = get_time_ns();
        uint64_t c0 = rdtscp();
        volatile int32_t tern_sink = 0;
        for (int i = 0; i < BENCH_ITERS; i++) {
            tern_sink += ternary_dot(ta, tb, n);
        }
        uint64_t c1 = rdtscp();
        double t1 = get_time_ns();
        (void)tern_sink;

        double tern_ns = (t1 - t0) / BENCH_ITERS;
        double tern_cyc = (double)(c1 - c0) / BENCH_ITERS;
        double tern_mdps = 1e3 / tern_ns;  /* Mdot/s = 1e9 / (ns * 1e6) = 1e3/ns */

        /* benchmark float32 */
        t0 = get_time_ns();
        c0 = rdtscp();
        volatile float fp32_sink = 0;
        for (int i = 0; i < BENCH_ITERS; i++) {
            fp32_sink += float32_dot(fa, fb, n);
        }
        c1 = rdtscp();
        t1 = get_time_ns();
        (void)fp32_sink;

        double fp32_ns = (t1 - t0) / BENCH_ITERS;
        double fp32_cyc = (double)(c1 - c0) / BENCH_ITERS;
        double fp32_mdps = 1e3 / fp32_ns;

        double speedup = fp32_ns / tern_ns;

        printf("%-8d | %-14.2f %-14.1f %-10.1f | %-14.2f %-14.1f %-10.1f | %-8.2fx | %s\n",
               n, tern_ns, tern_cyc, tern_mdps,
               fp32_ns, fp32_cyc, fp32_mdps,
               speedup, match ? "YES" : "NO");
    }

    printf("\n");
    printf("Notes:\n");
    printf("  Ternary: _mm256_sign_epi8 (32-wide int8, 1 cycle) + _mm256_add_epi8\n");
    printf("  Float32: _mm256_mul_ps (8-wide float, 1 cycle) + _mm256_add_ps\n");
    printf("  Ternary processes 32 elements/instruction vs Float32's 8 elements/instruction\n");
    printf("  Ternary uses 1 byte/element vs Float32's 4 bytes/element (4x memory advantage)\n");
    printf("  'Match?' verifies ternary result == (int32_t)float32 result (correctness check)\n");

    free(ta);
    free(tb);
    free(fa);
    free(fb);

    return 0;
}
