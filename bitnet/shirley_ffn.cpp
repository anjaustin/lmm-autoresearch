/*
 * shirley_ffn.cpp — Shirley FFN compute implementation
 *
 * Compiled with -mavx2 -mfma (or -march=native).
 * Calls ternary matmul kernels and Shirley integer kernels directly.
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

/* C++ compat for C headers using restrict */
#define restrict

#include "shirley_kernels.h"

/* Need full ggml_tensor definition */
#include "3rdparty/llama.cpp/ggml/include/ggml.h"
#include "shirley_ffn.h"

/* Ternary matmul kernel (defined in ggml-bitnet-mad.cpp) */
extern "C" {
void ggml_gemv_i2_i8_s(int n, float * s, size_t bs,
    const void * vx, const void * vy, int nr, int nc);
}

/* ================================================================
 *  Helper: float array → int8 at ±range
 * ================================================================ */
static float quantize_f32_to_i8(
    int8_t * dst, const float * src, int n, int range
) {
    float max_abs = 0.0f;
    for (int i = 0; i < n; i++) {
        float a = src[i] > 0 ? src[i] : -src[i];
        if (a > max_abs) max_abs = a;
    }
    if (max_abs < 1e-10f) {
        memset(dst, 0, n);
        return 0.0f;
    }
    float qs = (float)range / max_abs;
    for (int i = 0; i < n; i++) {
        int32_t r = (int32_t)(src[i] * qs + (src[i] >= 0 ? 0.5f : -0.5f));
        if (r > range) r = range;
        if (r < -range) r = -range;
        dst[i] = (int8_t)r;
    }
    return max_abs / (float)range;
}

/* ================================================================
 *  FFN compute — the whole block in one function
 * ================================================================ */

extern "C"
void shirley_ffn_compute(
    struct ggml_tensor * dst,
    const struct ggml_tensor * a,
    int ith, int nth,
    void * userdata
) {
    if (ith != 0) return;

    struct shirley_ffn_params * p = (struct shirley_ffn_params *)userdata;
    const int n = p->n_embd;
    const int n_ff = p->n_ff;
    const float * input = (const float *)a->data;

    /* 1. RMSNorm (ffn_norm): f32 → i8 */
    float act_scale = shirley_rmsnorm_quantize(
        p->w_act, input, p->ffn_norm_gamma, n, p->eps, 80);

    /* 2. Gate matmul: i8 × 2bit → raw int32 (in float) */
    ggml_gemv_i2_i8_s(n, p->w_raw, n_ff, p->gate_data, p->w_act, n_ff, 1);

    { static int dbg = 0; if (dbg < 1) {
        fprintf(stderr, "shirley: gate raw[0..3]=%.1f,%.1f,%.1f,%.1f act[0..3]=%d,%d,%d,%d\n",
            p->w_raw[0], p->w_raw[1], p->w_raw[2], p->w_raw[3],
            p->w_act[0], p->w_act[1], p->w_act[2], p->w_act[3]);
        dbg++; } }

    /* Rescale gate: raw → real → int8 */
    float gate_cs = p->gate_wscale * p->gate_lscale / act_scale;
    for (int i = 0; i < n_ff; i++) p->w_raw[i] *= gate_cs;
    quantize_f32_to_i8(p->w_gate, p->w_raw, n_ff, 80);

    /* 3. Up matmul: i8 × 2bit → raw → int8 */
    ggml_gemv_i2_i8_s(n, p->w_raw, n_ff, p->up_data, p->w_act, n_ff, 1);
    float up_cs = p->up_wscale * p->up_lscale / act_scale;
    for (int i = 0; i < n_ff; i++) p->w_raw[i] *= up_cs;
    quantize_f32_to_i8(p->w_up, p->w_raw, n_ff, 80);

    /* 4. ReLU(gate) */
    shirley_relu_i8(p->w_gate, p->w_gate, n_ff);

    /* 5. Square(ReLU(gate)) → int16 */
    shirley_square_i8_to_i16(p->w_sq, p->w_gate, n_ff);

    /* 6. gate² × up → int32 → int8 */
    {
        int32_t max_abs = 0;
        for (int i = 0; i < n_ff; i++) {
            p->w_prod[i] = (int32_t)p->w_sq[i] * (int32_t)p->w_up[i];
            int32_t av = p->w_prod[i] > 0 ? p->w_prod[i] : -p->w_prod[i];
            if (av > max_abs) max_abs = av;
        }
        if (max_abs > 0) {
            int64_t half = (int64_t)max_abs / 2;
            for (int i = 0; i < n_ff; i++) {
                int64_t v = (int64_t)p->w_prod[i] * 80;
                int32_t r;
                if (v >= 0) r = (int32_t)((v + half) / max_abs);
                else        r = (int32_t)(-((-v + half) / max_abs));
                if (r > 80) r = 80;
                if (r < -80) r = -80;
                p->w_ffn_out[i] = (int8_t)r;
            }
        } else {
            memset(p->w_ffn_out, 0, n_ff);
        }
    }

    /* 7. Sub-norm (ffn_sub_norm): i8 → float → norm → i8
     * Convert to float for shirley_rmsnorm_quantize.
     * TODO: use shirley_rmsnorm_ternary with Q14 gamma for zero-float path. */
    {
        /* Use w_raw as temp float buffer (n_ff fits since w_raw is max(n,n_ff)) */
        for (int i = 0; i < n_ff; i++) p->w_raw[i] = (float)p->w_ffn_out[i];
        act_scale = shirley_rmsnorm_quantize(
            p->w_sub, p->w_raw, p->ffn_sub_norm_gamma, n_ff, p->eps, 80);
    }

    /* 8. Down matmul: i8 × 2bit → raw → float */
    ggml_gemv_i2_i8_s(n_ff, p->w_raw, n, p->down_data, p->w_sub, n, 1);

    /* 9. Rescale + residual ADD → float output */
    float down_cs = p->down_wscale * p->down_lscale / act_scale;
    float * output = (float *)dst->data;
    for (int i = 0; i < n; i++) {
        output[i] = p->w_raw[i] * down_cs + input[i];
    }

    static int logged = 0;
    if (!logged) {
        fprintf(stderr, "shirley: FFN active (layer %d) act_scale=%.4f gate_raw[0..3]=%.1f,%.1f,%.1f,%.1f "
                "down_raw[0..3]=%.1f,%.1f,%.1f,%.1f out[0..3]=%.4f,%.4f,%.4f,%.4f\n",
                p->layer_idx, act_scale,
                p->w_raw[0], p->w_raw[1], p->w_raw[2], p->w_raw[3],
                0.0f, 0.0f, 0.0f, 0.0f, /* down_raw already overwritten */
                output[0], output[1], output[2], output[3]);
        logged = 1;
    }
}

/* ================================================================
 *  Initialization
 * ================================================================ */

extern "C"
void shirley_ffn_params_init(
    struct shirley_ffn_params * p,
    int n_embd, int n_ff, float eps, int layer_idx,
    const struct ggml_tensor * gate, const struct ggml_tensor * gate_scale_t,
    const struct ggml_tensor * up,   const struct ggml_tensor * up_scale_t,
    const struct ggml_tensor * down, const struct ggml_tensor * down_scale_t,
    const struct ggml_tensor * ffn_norm,
    const struct ggml_tensor * ffn_sub_norm
) {
    p->n_embd = n_embd;
    p->n_ff = n_ff;
    p->eps = eps;
    p->layer_idx = layer_idx;

    p->gate_data = gate->data;
    p->up_data = up->data;
    p->down_data = down->data;

    /* Weight scales: stored after packed 2-bit weights */
    p->gate_wscale = *(const float *)((const uint8_t *)gate->data + (gate->ne[0] * gate->ne[1] / 4));
    p->up_wscale = *(const float *)((const uint8_t *)up->data + (up->ne[0] * up->ne[1] / 4));
    p->down_wscale = *(const float *)((const uint8_t *)down->data + (down->ne[0] * down->ne[1] / 4));

    p->gate_lscale = gate_scale_t ? *(const float *)gate_scale_t->data : 1.0f;
    p->up_lscale = up_scale_t ? *(const float *)up_scale_t->data : 1.0f;
    p->down_lscale = down_scale_t ? *(const float *)down_scale_t->data : 1.0f;

    p->ffn_norm_gamma = ffn_norm ? (const float *)ffn_norm->data : NULL;
    p->ffn_sub_norm_gamma = ffn_sub_norm ? (const float *)ffn_sub_norm->data : NULL;

    int max_dim = n_embd > n_ff ? n_embd : n_ff;
    p->w_act     = (int8_t  *)malloc(max_dim * sizeof(int8_t));
    p->w_gate    = (int8_t  *)malloc(n_ff * sizeof(int8_t));
    p->w_up      = (int8_t  *)malloc(n_ff * sizeof(int8_t));
    p->w_ffn_out = (int8_t  *)malloc(n_ff * sizeof(int8_t));
    p->w_sub     = (int8_t  *)malloc(n_ff * sizeof(int8_t));
    p->w_raw     = (float   *)malloc(max_dim * sizeof(float));
    p->w_sq      = (int16_t *)malloc(n_ff * sizeof(int16_t));
    p->w_prod    = (int32_t *)malloc(n_ff * sizeof(int32_t));

    p->ready = 1;

    fprintf(stderr, "shirley: FFN params init layer %d (n=%d, n_ff=%d, "
            "ws=%.4f/%.4f/%.4f, ls=%.4f/%.4f/%.4f)\n",
            layer_idx, n_embd, n_ff,
            p->gate_wscale, p->up_wscale, p->down_wscale,
            p->gate_lscale, p->up_lscale, p->down_lscale);
}
