/*
 * shirley_ffn.h — Shirley FFN: interface declarations
 *
 * The compute function and initialization are in shirley_ffn.cpp,
 * compiled separately with AVX2 flags.
 */

#ifndef SHIRLEY_FFN_H
#define SHIRLEY_FFN_H

#include <stdint.h>
/* ggml.h is on the include path from cmake */
struct ggml_tensor;

struct shirley_ffn_params {
    int n_embd;
    int n_ff;
    float eps;
    int32_t eps_mant;
    int8_t  eps_exp;
    int layer_idx;

    const void * gate_data;
    const void * up_data;
    const void * down_data;

    float gate_wscale;
    float up_wscale;
    float down_wscale;

    float gate_lscale;
    float up_lscale;
    float down_lscale;

    /* Gamma weights — precomputed as MTFP21 at model load */
    int32_t * ffn_norm_gamma_mant;
    int8_t  * ffn_norm_gamma_exp;
    int32_t * ffn_sub_norm_gamma_mant;
    int8_t  * ffn_sub_norm_gamma_exp;

    int8_t  * w_act;
    int8_t  * w_gate;
    int8_t  * w_up;
    int8_t  * w_ffn_out;
    int8_t  * w_sub;
    float   * w_raw;
    int16_t * w_sq;
    int32_t * w_prod;

    /* Threading: shared workspace for multi-threaded matmul.
     * Thread 0 prepares activations, all threads execute matmul rows. */
    volatile int mt_phase;       /* atomic: 0=prep, 1=gate_up_ready, 2=gate_up_done, 3=down_ready, 4=down_done */
    int16_t * mt_act;            /* shared block-aligned activations */
    int8_t    mt_bexp;
    void    * mt_gate;           /* mtfp21_t[n_ff] — opaque for header */
    void    * mt_up;             /* mtfp21_t[n_ff] */
    void    * mt_down;           /* mtfp21_t[n_embd] */
    int16_t * mt_sub_act;
    int8_t    mt_sub_bexp;
    volatile int mt_threads_done;

    int ready;
};

#ifdef __cplusplus
extern "C" {
#endif

void shirley_ffn_compute(
    struct ggml_tensor * dst,
    const struct ggml_tensor * a,
    int ith, int nth,
    void * userdata);

void shirley_ffn_params_init(
    struct shirley_ffn_params * p,
    int n_embd, int n_ff, float eps, int layer_idx,
    const struct ggml_tensor * gate, const struct ggml_tensor * gate_scale_t,
    const struct ggml_tensor * up,   const struct ggml_tensor * up_scale_t,
    const struct ggml_tensor * down, const struct ggml_tensor * down_scale_t,
    const struct ggml_tensor * ffn_norm,
    const struct ggml_tensor * ffn_sub_norm);

#ifdef __cplusplus
}
#endif

#endif /* SHIRLEY_FFN_H */
