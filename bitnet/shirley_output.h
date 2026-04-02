/*
 * shirley_output.h — Output norm in MTFP21
 *
 * Replaces the final ggml RMSNorm after the layer loop.
 * Input is float32 (from last layer's FFN output).
 * Output is float32 (for the LM head matmul).
 * Internally, the norm is computed in MTFP21 for consistency.
 */

#ifndef SHIRLEY_OUTPUT_H
#define SHIRLEY_OUTPUT_H

#include <stdint.h>

struct ggml_tensor;

struct shirley_output_params {
    int n_embd;
    float eps;
    const float * output_norm_gamma;  /* [n_embd] */
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

void shirley_output_params_init(
    struct shirley_output_params * p,
    int n_embd, float eps,
    const struct ggml_tensor * output_norm);

#ifdef __cplusplus
}
#endif

#endif
