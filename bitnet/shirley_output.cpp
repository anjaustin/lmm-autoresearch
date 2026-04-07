/*
 * shirley_output.cpp — Output norm + MTFP10 LM head
 *
 * Output norm: float → shirley_rmsnorm_quantize → int8 (reuses the kernel)
 * LM head: int16 hidden × int16 embedding → int32 → float logits
 *
 * The embedding table is converted from f16 to block-aligned MTFP10
 * (int16 mantissa, per-row int8 exponent) at model load. Same memory
 * footprint as f16. Integer compute via _mm256_madd_epi16.
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <immintrin.h>

#define restrict
#include "shirley_kernels.h"
#include "shirley_convert.h"

#include "3rdparty/llama.cpp/ggml/include/ggml.h"
#include "shirley_output.h"
#include "shirley_barrier.h"

/* ================================================================
 *  MTFP10 block-align: float row → int16 mantissas + block exponent
 *
 *  MTFP10: 10-trit mantissa, range ±29524, fits int16.
 *  Block exponent: shared per row, value = mantissa × 3^exponent.
 *  Precision: 15.8 bits (exceeds f16's 11-bit mantissa).
 * ================================================================ */

#define MTFP10_MANT_MAX 29524  /* (3^10 - 1) / 2 */

static void convert_f32_row_to_mtfp10(
    int16_t * dst_mant, int8_t * dst_exp,
    const float * src, int n
) {
    /* Find max absolute value */
    float max_abs = 0.0f;
    for (int i = 0; i < n; i++) {
        float a = fabsf(src[i]);
        if (a > max_abs) max_abs = a;
    }

    if (max_abs == 0.0f) {
        memset(dst_mant, 0, n * sizeof(int16_t));
        *dst_exp = 0;
        return;
    }

    /* Find block exponent: max_abs ≈ MTFP10_MANT_MAX × 3^exp
     * exp = log3(max_abs / MTFP10_MANT_MAX) */
    float scale = (float)MTFP10_MANT_MAX / max_abs;
    int8_t exp = 0;

    /* Compute in base-3: if scale < 1, we need positive exponent (values are large)
     * if scale > 1, we need negative exponent (values are small) */
    float test = max_abs;
    if (test > MTFP10_MANT_MAX) {
        while (test > MTFP10_MANT_MAX) { test /= 3.0f; exp++; }
    } else {
        while (test * 3.0f <= MTFP10_MANT_MAX) { test *= 3.0f; exp--; }
    }

    /* Quantize: each element = round(src[i] / 3^exp) clamped to ±MTFP10_MANT_MAX */
    float divisor = powf(3.0f, (float)exp);
    float inv_div = 1.0f / divisor;
    for (int i = 0; i < n; i++) {
        float scaled = src[i] * inv_div;
        int32_t q = (int32_t)roundf(scaled);
        if (q > MTFP10_MANT_MAX) q = MTFP10_MANT_MAX;
        if (q < -MTFP10_MANT_MAX) q = -MTFP10_MANT_MAX;
        dst_mant[i] = (int16_t)q;
    }
    *dst_exp = exp;
}

/* f16 → float helper (ggml stores f16 as ggml_fp16_t which is uint16_t) */
static inline float fp16_to_f32(uint16_t h) {
    /* Use hardware conversion via _mm_cvtph_ps if available */
    __m128i v = _mm_set1_epi16((short)h);
    __m128 f = _mm_cvtph_ps(v);
    return _mm_cvtss_f32(f);
}

/* ================================================================
 *  Output norm: shirley_rmsnorm_quantize (float → float normed)
 *  Simpler than attention/FFN: just norm, no quantize to int8 here.
 *  The LM head needs float input for block-alignment.
 * ================================================================ */

extern "C"
void shirley_output_compute(
    struct ggml_tensor * dst,
    const struct ggml_tensor * a,
    int ith, int nth,
    void * userdata
) {
    if (ith != 0) return;

    struct shirley_output_params * p = (struct shirley_output_params *)userdata;
    const int n = p->n_embd;
    const int n_tokens = (int)a->ne[1];
    float * output = (float *)dst->data;

    for (int tok = 0; tok < n_tokens; tok++) {
        const float * input = (const float *)a->data + tok * n;
        float * out_tok = output + tok * n;

        /* RMSNorm in float — the output feeds the MTFP10 LM head */
        float sum_sq = 0.0f;
        for (int i = 0; i < n; i++) sum_sq += input[i] * input[i];
        float scale = 1.0f / sqrtf(sum_sq / n + p->eps);

        if (p->output_norm_gamma_f32) {
            for (int i = 0; i < n; i++)
                out_tok[i] = input[i] * scale * p->output_norm_gamma_f32[i];
        } else {
            for (int i = 0; i < n; i++)
                out_tok[i] = input[i] * scale;
        }
    }

    static int logged = 0;
    if (!logged) {
        fprintf(stderr, "shirley: float output norm active (%d tokens)\n", n_tokens);
        logged = 1;
    }
}

/* ================================================================
 *  LM head: MTFP10 int16 GEMV — multi-threaded
 *
 *  Each thread processes a partition of vocab rows.
 *  Per row: madd_epi16 dot product (int16 × int16 → int32),
 *  then convert to float logit using block exponents.
 * ================================================================ */

/* Single dot product: int16 activation × int16 embedding row → int32 */
static inline int32_t dot_i16_avx2(
    const int16_t * restrict a,
    const int16_t * restrict b,
    int n
) {
    __m256i acc = _mm256_setzero_si256();
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        __m256i va = _mm256_loadu_si256((const __m256i *)(a + i));
        __m256i vb = _mm256_loadu_si256((const __m256i *)(b + i));
        /* madd_epi16: multiply adjacent int16 pairs, sum to int32 */
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(va, vb));
    }
    /* Horizontal sum */
    __m128i lo = _mm256_castsi256_si128(acc);
    __m128i hi = _mm256_extracti128_si256(acc, 1);
    __m128i sum4 = _mm_add_epi32(lo, hi);
    sum4 = _mm_hadd_epi32(sum4, sum4);
    sum4 = _mm_hadd_epi32(sum4, sum4);
    int32_t result = _mm_cvtsi128_si32(sum4);

    /* Scalar tail */
    for (; i < n; i++) result += (int32_t)a[i] * (int32_t)b[i];
    return result;
}

extern "C"
void shirley_lmhead_compute(
    struct ggml_tensor * dst,
    const struct ggml_tensor * a,   /* shape template (unused data) */
    const struct ggml_tensor * b,   /* normed output [n_embd, n_tokens] */
    int ith, int nth,
    void * userdata
) {
    struct shirley_output_params * p = (struct shirley_output_params *)userdata;
    const int n = p->n_embd;
    const int V = p->vocab_size;
    const int n_tokens = (int)b->ne[1];
    float * output = (float *)dst->data;

    /* Lazy conversion: convert embedding table on first call */
    if (!p->embd_mtfp10 && ith == 0) {
        const struct ggml_tensor * te = (const struct ggml_tensor *)p->_tok_embd_tensor;
        if (te && te->data) {
            fprintf(stderr, "shirley: converting embedding table to MTFP10 (%d × %d)...\n", V, n);
            p->embd_mtfp10 = (int16_t *)malloc((int64_t)V * n * sizeof(int16_t));
            p->embd_row_exp = (int8_t *)malloc(V * sizeof(int8_t));

            const void * raw_data = te->data;
            int src_type = te->type; /* 0=f32, 1=f16 */

            for (int v = 0; v < V; v++) {
                float row_f32[n]; /* VLA */
                if (src_type == 1) { /* f16 */
                    const uint16_t * src_f16 = (const uint16_t *)raw_data + (int64_t)v * n;
                    for (int i = 0; i < n; i++) row_f32[i] = fp16_to_f32(src_f16[i]);
                } else { /* f32 */
                    const float * src_f32 = (const float *)raw_data + (int64_t)v * n;
                    memcpy(row_f32, src_f32, n * sizeof(float));
                }
                convert_f32_row_to_mtfp10(
                    p->embd_mtfp10 + (int64_t)v * n,
                    &p->embd_row_exp[v],
                    row_f32, n);
            }
            fprintf(stderr, "shirley: MTFP10 embedding ready (%.1f MB)\n",
                    (float)((int64_t)V * n * 2 + V) / (1024.0f * 1024.0f));
        }
    }

    /* Barrier: ensure conversion is complete before any thread reads */
    if (nth > 1) {
        __atomic_fetch_add(&p->mt_threads_done, 1, __ATOMIC_ACQ_REL);
        if (ith == 0) {
            while (__atomic_load_n(&p->mt_threads_done, __ATOMIC_ACQUIRE) < nth) { _mm_pause(); }
            __atomic_store_n(&p->mt_threads_done, 0, __ATOMIC_RELEASE);
            __atomic_store_n(&p->mt_phase, 1, __ATOMIC_RELEASE);
        } else {
            while (__atomic_load_n(&p->mt_phase, __ATOMIC_ACQUIRE) < 1) { _mm_pause(); }
        }
    }

    if (!p->embd_mtfp10) {
        /* Fallback: use float dot product */
        if (ith != 0) return;
        const struct ggml_tensor * te = (const struct ggml_tensor *)p->_tok_embd_tensor;
        if (!te || !te->data) { memset(output, 0, (int64_t)V * n_tokens * sizeof(float)); return; }
        const float * embd_f32 = (const float *)te->data;

        for (int tok = 0; tok < n_tokens; tok++) {
            const float * inp = (const float *)b->data + tok * n;
            float * out_tok = output + tok * V;
            for (int v = 0; v < V; v++) {
                float dot = 0.0f;
                const float * row = embd_f32 + (int64_t)v * n;
                for (int d = 0; d < n; d++) dot += inp[d] * row[d];
                out_tok[v] = dot;
            }
        }
        return;
    }

    /* ---- MTFP10 LM head: int16 × int16 → int32 → float logits ---- */

    for (int tok = 0; tok < n_tokens; tok++) {
        const float * inp = (const float *)b->data + tok * n;
        float * out_tok = output + tok * V;

        /* Block-align the hidden state to int16 (same format as embedding rows) */
        int16_t act_i16[n]; /* VLA */
        float act_max = 0.0f;
        for (int i = 0; i < n; i++) {
            float a = fabsf(inp[i]);
            if (a > act_max) act_max = a;
        }

        int8_t act_exp = 0;
        if (act_max > 0.0f) {
            float test = act_max;
            if (test > MTFP10_MANT_MAX) {
                while (test > MTFP10_MANT_MAX) { test /= 3.0f; act_exp++; }
            } else {
                while (test * 3.0f <= MTFP10_MANT_MAX) { test *= 3.0f; act_exp--; }
            }
            float inv_div = 1.0f / powf(3.0f, (float)act_exp);
            for (int i = 0; i < n; i++) {
                int32_t q = (int32_t)roundf(inp[i] * inv_div);
                if (q > MTFP10_MANT_MAX) q = MTFP10_MANT_MAX;
                if (q < -MTFP10_MANT_MAX) q = -MTFP10_MANT_MAX;
                act_i16[i] = (int16_t)q;
            }
        } else {
            memset(act_i16, 0, n * sizeof(int16_t));
        }

        /* Partition vocab rows across threads */
        int rows_per = (V + nth - 1) / nth;
        int r0 = ith * rows_per;
        int r1 = r0 + rows_per; if (r1 > V) r1 = V;

        /* Precompute the base-3 power for combined exponent recovery */
        /* logit = dot_int32 × 3^(act_exp + row_exp) */
        for (int v = r0; v < r1; v++) {
            int32_t dot = dot_i16_avx2(act_i16,
                p->embd_mtfp10 + (int64_t)v * n, n);
            int combined_exp = (int)act_exp + (int)p->embd_row_exp[v];
            float logit = (float)dot * powf(3.0f, (float)combined_exp);
            out_tok[v] = logit;
        }
    }

    /* End-of-token sync */
    if (nth > 1) {
        __atomic_fetch_add(&p->mt_threads_done, 1, __ATOMIC_ACQ_REL);
        if (ith == 0) {
            while (__atomic_load_n(&p->mt_threads_done, __ATOMIC_ACQUIRE) < nth) { _mm_pause(); }
            __atomic_store_n(&p->mt_threads_done, 0, __ATOMIC_RELEASE);
            __atomic_store_n(&p->mt_phase, 0, __ATOMIC_RELEASE);
        } else {
            while (__atomic_load_n(&p->mt_phase, __ATOMIC_ACQUIRE) != 0) { _mm_pause(); }
        }
    }

    static int logged = 0;
    if (!logged && ith == 0) {
        fprintf(stderr, "shirley: MTFP10 LM head active (%d tokens, vocab=%d, %d threads)\n",
                n_tokens, V, nth);
        logged = 1;
    }
}

/* ================================================================
 *  Initialization
 * ================================================================ */

extern "C"
void shirley_output_params_init(
    struct shirley_output_params * p,
    int n_embd, int vocab_size, float eps,
    const struct ggml_tensor * output_norm,
    const struct ggml_tensor * tok_embd
) {
    p->n_embd = n_embd;
    p->vocab_size = vocab_size;
    p->eps = eps;
    { mtfp21_t e = mtfp21_from_float(eps); p->eps_mant = e.mantissa; p->eps_exp = e.exponent; }
    shirley_convert_f32_to_mtfp21(
        &p->output_norm_gamma_mant, &p->output_norm_gamma_exp,
        output_norm ? (const float *)output_norm->data : NULL, n_embd);
    p->output_norm_gamma_f32 = output_norm ? (const float *)output_norm->data : NULL;

    /* Embedding table: defer conversion to first compute call
     * (tensor data may not be loaded yet at init time) */
    p->embd_mtfp10 = NULL;
    p->embd_row_exp = NULL;
    p->embd_mtfp21 = NULL;
    p->_tok_embd_tensor = (const void *)tok_embd;

    p->mt_phase = 0;
    p->mt_threads_done = 0;

    p->ready = 1;

    fprintf(stderr, "shirley: output init (n=%d, vocab=%d, MTFP10 LM head)\n",
            n_embd, vocab_size);
}
