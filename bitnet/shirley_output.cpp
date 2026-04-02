/*
 * shirley_output.cpp — Output norm in MTFP21
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

        /* Convert to MTFP21 */
        mtfp21_t inp_m[n]; /* VLA */
        for (int i = 0; i < n; i++) inp_m[i] = mtfp21_from_float(input[i]);

        /* RMSNorm in MTFP21 */
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
            /* Output as float for LM head matmul */
            out_tok[i] = mtfp21_to_float(normed);
        }
    }

    static int logged = 0;
    if (!logged) {
        fprintf(stderr, "shirley: MTFP21 output norm active (%d tokens)\n", n_tokens);
        logged = 1;
    }
}

extern "C"
void shirley_output_params_init(
    struct shirley_output_params * p,
    int n_embd, float eps,
    const struct ggml_tensor * output_norm
) {
    p->n_embd = n_embd;
    p->eps = eps;
    p->output_norm_gamma = output_norm ? (const float *)output_norm->data : NULL;
    p->ready = 1;
    fprintf(stderr, "shirley: MTFP21 output norm init (n=%d)\n", n_embd);
}
