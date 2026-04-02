/*
 * shirley_output.h — Output norm + LM head, fully MTFP21
 *
 * Replaces: output RMSNorm + embedding matmul (LM head)
 * Embedding table converted to MTFP21 at model load.
 * Output is float32 logits (for sampling — the final boundary).
 */

#ifndef SHIRLEY_OUTPUT_H
#define SHIRLEY_OUTPUT_H

#include <stdint.h>

struct ggml_tensor;
struct mtfp21_t_fwd;  /* forward decl */

struct shirley_output_params {
    int n_embd;
    int vocab_size;
    float eps;
    const float * output_norm_gamma;  /* [n_embd] */

    /* Embedding table converted to MTFP21 at model load.
     * Layout: [vocab_size][n_embd] as mtfp21_t structs.
     * Each entry is a position in the geometric space. */
    void * embd_mtfp21;   /* mtfp21_t[vocab_size * n_embd] */

    int ready;
};

#ifdef __cplusplus
extern "C" {
#endif

void shirley_output_compute(
    struct ggml_tensor * dst,
    const struct ggml_tensor * a,
    int ith, int nth,
    void * userdata);

void shirley_lmhead_compute(
    struct ggml_tensor * dst,
    const struct ggml_tensor * a,
    const struct ggml_tensor * b,
    int ith, int nth,
    void * userdata);

void shirley_output_params_init(
    struct shirley_output_params * p,
    int n_embd, int vocab_size, float eps,
    const struct ggml_tensor * output_norm,
    const struct ggml_tensor * tok_embd);

#ifdef __cplusplus
}
#endif

#endif
