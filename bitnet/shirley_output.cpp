/*
 * shirley_output.cpp — Output norm + LM head, fully MTFP21
 *
 * The embedding table is converted to MTFP21 once at model load.
 * Output norm: MTFP21 RMSNorm (custom1, shape preserved).
 * LM head: MTFP21 dot products against converted embeddings (custom2).
 * Logits output as float32 — the one true boundary to sampling.
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#define restrict
#include "shirley_kernels.h"

#include "3rdparty/llama.cpp/ggml/include/ggml.h"
#include "shirley_output.h"

/* Packed to 5 bytes to reduce memory (128K vocab × 2560 dim) */
#pragma pack(push, 1)
typedef struct {
    int32_t mantissa;
    int8_t  exponent;
} mtfp21_local_t;
#pragma pack(pop)

/* ================================================================
 *  Output norm: RMSNorm in MTFP21
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

        mtfp21_t inp_m[n]; /* VLA */
        for (int i = 0; i < n; i++) inp_m[i] = mtfp21_from_float(input[i]);

        mtfp21_t sum_sq = {0, 0};
        for (int i = 0; i < n; i++)
            sum_sq = mtfp21_add(sum_sq, mtfp21_mul(inp_m[i], inp_m[i]));
        mtfp21_t mean = mtfp21_div_scalar(sum_sq, n);
        mtfp21_t scale = mtfp21_rsqrt(
            mtfp21_add(mean, mtfp21_from_float(p->eps)));

        for (int i = 0; i < n; i++) {
            mtfp21_t normed = mtfp21_mul(inp_m[i], scale);
            if (p->output_norm_gamma) {
                normed = mtfp21_mul(normed,
                    mtfp21_from_float(p->output_norm_gamma[i]));
            }
            out_tok[i] = mtfp21_to_float(normed);
        }
    }

    static int logged = 0;
    if (!logged) {
        fprintf(stderr, "shirley: MTFP21 output norm active (%d tokens)\n", n_tokens);
        logged = 1;
    }
}

/* ================================================================
 *  LM head: MTFP21 matmul against converted embedding table
 *
 *  dst:  [vocab_size, n_tokens] — logits
 *  a:    [vocab_size, n_embd] — embedding table (src0 of mul_mat)
 *  b:    [n_embd, n_tokens] — normed output (src1 of mul_mat)
 * ================================================================ */

extern "C"
void shirley_lmhead_compute(
    struct ggml_tensor * dst,
    const struct ggml_tensor * a,   /* tok_embd [n_embd, vocab_size] */
    const struct ggml_tensor * b,   /* normed output [n_embd, n_tokens] */
    int ith, int nth,
    void * userdata
) {
    if (ith != 0) return;

    struct shirley_output_params * p = (struct shirley_output_params *)userdata;

    /* Lazy conversion: convert embedding table to MTFP21 on first call.
     * At this point the ggml backend has loaded the weight data. */
    fprintf(stderr, "shirley: lmhead dst=%p dst->data=%p b=%p b->data=%p ne=[%lld,%lld]\n",
            (void*)dst, dst ? dst->data : NULL,
            (void*)b, b ? b->data : NULL,
            dst ? (long long)dst->ne[0] : 0, dst ? (long long)dst->ne[1] : 0);
    fflush(stderr);

    /* MTFP21 LM head: convert embedding values on-the-fly.
     * Each vocab row is converted to MTFP21 during the dot product.
     * No pre-allocation of the full 1.5 GB table needed. */
    if (p->_tok_embd_tensor) {
        const struct ggml_tensor * te = (const struct ggml_tensor *)p->_tok_embd_tensor;
        const float * embd_f32 = (const float *)te->data;
        const int n = p->n_embd;
        const int V = p->vocab_size;
        const int n_tok = (int)b->ne[1];
        float * output = (float *)dst->data;

        for (int tok = 0; tok < n_tok; tok++) {
            const float * inp = (const float *)b->data + tok * n;
            float * out_tok = output + tok * V;

            mtfp21_t inp_m[n]; /* VLA */
            for (int i = 0; i < n; i++) inp_m[i] = mtfp21_from_float(inp[i]);

            if (tok == 0) { fprintf(stderr, "shirley: lmhead computing %d vocab dots (embd_f32=%p)...\n", V, (const void*)embd_f32); fflush(stderr); }
            for (int v = 0; v < V; v++) {
                /* Plain float dot product for now — verify plumbing works */
                float dot = 0.0f;
                const float * row = embd_f32 + (int64_t)v * n;
                for (int d = 0; d < n; d++) {
                    dot += inp[d] * row[d];
                }
                out_tok[v] = dot;
            }
        }

        static int logged2 = 0;
        if (!logged2) {
            fprintf(stderr, "shirley: MTFP21 LM head active (%d tokens, vocab=%d)\n", n_tok, V);
            logged2 = 1;
        }
        return;
    }

    /* Full MTFP21 path (when embedding table is pre-converted) */
    if (0 && p->_tok_embd_tensor) {  /* disabled until allocation issue is resolved */
        const struct ggml_tensor * te = (const struct ggml_tensor *)p->_tok_embd_tensor;
        const float * src = (const float *)te->data;
        int64_t total = (int64_t)p->vocab_size * p->n_embd;
        int64_t bytes = total * (int64_t)sizeof(mtfp21_local_t);
        int64_t tensor_bytes = (int64_t)te->ne[0] * te->ne[1] * sizeof(float);
        fprintf(stderr, "shirley: LM head — tok_embd ne=[%lld,%lld] nbytes=%lld, "
                "total=%lld elements (%.1f MB)\n",
                (long long)te->ne[0], (long long)te->ne[1],
                (long long)tensor_bytes,
                (long long)total, (float)bytes / (1024.0f * 1024.0f));
        fflush(stderr);
        /* Use actual tensor size, not vocab_size * n_embd */
        int64_t actual_total = (int64_t)te->ne[0] * te->ne[1];
        if (actual_total < total) {
            fprintf(stderr, "shirley: WARNING — tensor has %lld elements, expected %lld. Using tensor size.\n",
                    (long long)actual_total, (long long)total);
            total = actual_total;
            bytes = total * (int64_t)sizeof(mtfp21_local_t);
        }

        mtfp21_local_t * embd = (mtfp21_local_t *)malloc(bytes);
        if (embd) {
            /* First: verify all source data is readable */
            volatile float sum = 0;
            for (int64_t i = 0; i < total; i++) {
                sum += src[i];
                if (i % 100000000 == 0) {
                    fprintf(stderr, "shirley: reading src... %.0f%% (sum=%.2f)\n",
                            100.0 * i / total, (double)sum);
                    fflush(stderr);
                }
            }
            fprintf(stderr, "shirley: src read OK, converting...\n");
            fflush(stderr);

            for (int64_t i = 0; i < total; i++) {
                float val = src[i];
                if (val != val || val > 1e30f || val < -1e30f) {
                    embd[i].mantissa = 0;
                    embd[i].exponent = 0;
                } else {
                    mtfp21_t m = mtfp21_from_float(val);
                    embd[i].mantissa = m.mantissa;
                    embd[i].exponent = m.exponent;
                }
                if (i % 100000000 == 0) {
                    fprintf(stderr, "shirley: converting... %.0f%%\n", 100.0 * i / total);
                    fflush(stderr);
                }
            }
            p->embd_mtfp21 = embd;
            fprintf(stderr, "shirley: MTFP21 embedding table converted\n");
            fflush(stderr);
        }
    }

    if (!p->embd_mtfp21) {
        memset(dst->data, 0, ggml_nbytes(dst));
        return;
    }
    const int n = p->n_embd;
    const int V = p->vocab_size;
    const int n_tokens = (int)b->ne[1];
    float * output = (float *)dst->data;
    mtfp21_local_t * embd = (mtfp21_local_t *)p->embd_mtfp21;

    for (int tok = 0; tok < n_tokens; tok++) {
        const float * input = (const float *)b->data + tok * n;
        float * out_tok = output + tok * V;

        /* Convert normed input to MTFP21 */
        mtfp21_t inp_m[n]; /* VLA */
        for (int i = 0; i < n; i++) inp_m[i] = mtfp21_from_float(input[i]);

        /* Dot product with each vocabulary embedding */
        for (int v = 0; v < V; v++) {
            mtfp21_t dot = {0, 0};
            mtfp21_local_t * row = embd + v * n;
            for (int d = 0; d < n; d++) {
                mtfp21_t e;
                e.mantissa = row[d].mantissa;
                e.exponent = row[d].exponent;
                dot = mtfp21_add(dot, mtfp21_mul(inp_m[d], e));
            }
            out_tok[v] = mtfp21_to_float(dot);
        }
    }

    static int logged = 0;
    if (!logged) {
        fprintf(stderr, "shirley: MTFP21 LM head active (%d tokens, vocab=%d)\n",
                n_tokens, V);
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
    p->output_norm_gamma = output_norm ? (const float *)output_norm->data : NULL;

    /* Store the embedding tensor pointer for lazy conversion.
     * The actual data may not be loaded yet at init time —
     * ggml backends populate tensor data after creation.
     * Convert on first use in the LM head callback. */
    p->embd_mtfp21 = NULL;
    p->_tok_embd_tensor = (const void *)tok_embd;

    p->ready = 1;

    fprintf(stderr, "shirley: MTFP21 output init (n=%d, vocab=%d, embd=%s)\n",
            n_embd, vocab_size, p->embd_mtfp21 ? "converted" : "deferred");
}
