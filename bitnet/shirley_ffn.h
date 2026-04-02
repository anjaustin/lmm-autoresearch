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

    const float * ffn_norm_gamma;
    const float * ffn_sub_norm_gamma;

    int8_t  * w_act;
    int8_t  * w_gate;
    int8_t  * w_up;
    int8_t  * w_ffn_out;
    int8_t  * w_sub;
    float   * w_raw;
    int16_t * w_sq;
    int32_t * w_prod;

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
