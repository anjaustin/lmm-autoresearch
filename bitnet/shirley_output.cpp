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

typedef struct {
    int32_t mantissa;
    int8_t  exponent;
} mtfp21_local_t;

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

    /* Convert embedding table to MTFP21 — once, at model load */
    if (tok_embd) {
        int total = vocab_size * n_embd;
        mtfp21_local_t * embd = (mtfp21_local_t *)malloc(total * sizeof(mtfp21_local_t));
        const float * src = (const float *)tok_embd->data;
        for (int i = 0; i < total; i++) {
            mtfp21_t m = mtfp21_from_float(src[i]);
            embd[i].mantissa = m.mantissa;
            embd[i].exponent = m.exponent;
        }
        p->embd_mtfp21 = embd;
    } else {
        p->embd_mtfp21 = NULL;
    }

    p->ready = 1;

    fprintf(stderr, "shirley: MTFP21 output init (n=%d, vocab=%d, embd=%s)\n",
            n_embd, vocab_size, p->embd_mtfp21 ? "converted" : "deferred");
}
