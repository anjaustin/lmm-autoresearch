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
    int32_t eps_mant;
    int8_t  eps_exp;
    int32_t * output_norm_gamma_mant;  /* [n_embd] — precomputed MTFP21 */
    int8_t  * output_norm_gamma_exp;
    const float * output_norm_gamma_f32; /* float for shirley_rmsnorm_quantize */

    /* MTFP10 embedding table: block-aligned int16 mantissas + per-row exponent.
     * Converted from f16 at model load. Same memory footprint as f16.
     * Used by the LM head GEMV — int16 × int16 → int32 dot products. */
    int16_t * embd_mtfp10;   /* [vocab_size * n_embd] block-aligned int16 */
    int8_t  * embd_row_exp;  /* [vocab_size] per-row block exponent */

    /* Legacy fields — kept for compatibility */
    void * embd_mtfp21;
    const void * _tok_embd_tensor;

    /* Threading */
    volatile int mt_phase;
    volatile int mt_threads_done;

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
