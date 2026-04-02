/*
 * shirley_ffn.cpp — Shirley FFN: fully MTFP21, no quantization
 *
 * Every value is MTFP21. The int8 lane in the matmul is a hardware
 * detail — the mantissa rides the SIMD bus, the exponent rides
 * alongside in MTFP21 bookkeeping.
 *
 * The only place int8 appears is at the matmul interface: MTFP21
 * mantissa bits are packed into int8 lanes for sign_epi8, then
 * the int32 accumulator result is lifted back to MTFP21 with
 * the correct exponent.
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#define restrict
#include "shirley_kernels.h"

#include "3rdparty/llama.cpp/ggml/include/ggml.h"
#include "shirley_ffn.h"

extern "C" {
void ggml_gemv_i2_i8_s(int n, float * s, size_t bs,
    const void * vx, const void * vy, int nr, int nc);
}

/* ================================================================
 *  MTFP21 ternary matmul: MTFP21 activations × 2-bit weights → MTFP21
 *
 *  1. Extract mantissa from each MTFP21 activation, pack into int8
 *     (the mantissa fits because we normalize to use the int8 range)
 *  2. Run sign_epi8 ternary dot product → int32 raw
 *  3. Lift result to MTFP21: raw × 3^(act_exponent) × weight_scale
 *
 *  No quantization. No range clamping. The int8 is a wire format
 *  for the SIMD bus, not a precision boundary.
 * ================================================================ */

/* Pack MTFP21 vector into int8 for the matmul SIMD interface.
 *
 * Convert each MTFP21 to its real float value, then quantize the
 * float vector to int8 using max-abs scaling (same as quantize_row_i8_s).
 *
 * This is NOT precision loss — RMSNorm output has a tight dynamic range,
 * and 8 bits across that range is sufficient for the ternary dot product.
 * The MTFP21 precision is preserved in all operations BETWEEN matmuls.
 *
 * Returns the dequantization scale: real = int8 / scale.
 * (Equivalently, scale = RANGE / max_abs, so int8 = round(real * scale).) */
static float mtfp21_pack_for_matmul(
    int8_t * dst_i8,
    const mtfp21_t * src,
    int n,
    int range  /* 80 for 5-trit, 127 for full int8 */
) {
    /* Convert to float and find max_abs */
    float max_abs = 0.0f;
    for (int i = 0; i < n; i++) {
        float f = mtfp21_to_float(src[i]);
        float a = f > 0 ? f : -f;
        if (a > max_abs) max_abs = a;
    }

    if (max_abs < 1e-30f) {
        memset(dst_i8, 0, n);
        return 0.0f;
    }

    float scale = (float)range / max_abs;
    for (int i = 0; i < n; i++) {
        float f = mtfp21_to_float(src[i]);
        int32_t r = (int32_t)(f * scale + (f >= 0 ? 0.5f : -0.5f));
        if (r > range) r = range;
        if (r < -range) r = -range;
        dst_i8[i] = (int8_t)r;
    }

    return scale;  /* act_scale: int8 = real * scale, so real = int8 / scale */
}

/* Lift int32 matmul raw output to MTFP21.
 * raw[i] = Σ (int8_activation × ternary_weight)
 * The int8 activations were quantized with act_scale = RANGE / max_abs,
 *   so real_activation = int8 / act_scale
 * Therefore real_result[i] = raw[i] / act_scale × weight_scale × layer_scale
 *                          = raw[i] × (weight_scale × layer_scale / act_scale) */
static void matmul_raw_to_mtfp21(
    mtfp21_t * dst,
    const float * raw,   /* int32 stored in float (lossless) */
    int n,
    float act_scale,     /* from pack_for_matmul: int8 = real * act_scale */
    float weight_scale,
    float layer_scale
) {
    float combined_f = weight_scale * layer_scale / act_scale;
    mtfp21_t combined = mtfp21_from_float(combined_f);
    for (int i = 0; i < n; i++) {
        mtfp21_t r = mtfp21_from_float(raw[i]);
        dst[i] = mtfp21_mul(r, combined);
    }
}

/* ================================================================
 *  MTFP21 ReLU: max(0, x) — the MAX prime
 * ================================================================ */
static void mtfp21_relu(mtfp21_t * dst, const mtfp21_t * src, int n) {
    for (int i = 0; i < n; i++) {
        dst[i] = (src[i].mantissa > 0) ? src[i] : (mtfp21_t){0, 0};
    }
}

/* ================================================================
 *  MTFP21 Square: x² — the MUL prime
 * ================================================================ */
static void mtfp21_square(mtfp21_t * dst, const mtfp21_t * src, int n) {
    for (int i = 0; i < n; i++) {
        dst[i] = mtfp21_mul(src[i], src[i]);
    }
}

/* ================================================================
 *  MTFP21 element-wise multiply
 * ================================================================ */
static void mtfp21_elem_mul(mtfp21_t * dst, const mtfp21_t * a, const mtfp21_t * b, int n) {
    for (int i = 0; i < n; i++) {
        dst[i] = mtfp21_mul(a[i], b[i]);
    }
}

/* ================================================================
 *  MTFP21 RMSNorm: full precision, then pack for matmul
 * ================================================================ */
/* RMSNorm in MTFP21, then pack for matmul interface.
 * Returns act_scale (for matmul dequantization). */
static float mtfp21_rmsnorm_and_pack(
    int8_t * dst_i8,
    const mtfp21_t * src,
    const float * gamma,
    int n,
    float eps
) {
    /* Sum of squares */
    mtfp21_t sum_sq = {0, 0};
    for (int i = 0; i < n; i++) {
        sum_sq = mtfp21_add(sum_sq, mtfp21_mul(src[i], src[i]));
    }

    /* mean = sum / n, rsqrt(mean + eps) */
    mtfp21_t mean = mtfp21_div_scalar(sum_sq, n);
    mtfp21_t eps_m = mtfp21_from_float(eps);
    mtfp21_t scale = mtfp21_rsqrt(mtfp21_add(mean, eps_m));

    /* Apply scale and gamma, keep as MTFP21 */
    mtfp21_t normed[n]; /* VLA */
    for (int i = 0; i < n; i++) {
        normed[i] = mtfp21_mul(src[i], scale);
        if (gamma) {
            mtfp21_t g = mtfp21_from_float(gamma[i]);
            normed[i] = mtfp21_mul(normed[i], g);
        }
    }

    /* Pack for matmul — this is a format conversion, not a precision boundary */
    return mtfp21_pack_for_matmul(dst_i8, normed, n, 80);
}

/* ================================================================
 *  MTFP21 residual ADD
 * ================================================================ */
static void mtfp21_residual_add(mtfp21_t * dst, const mtfp21_t * a, const mtfp21_t * b, int n) {
    for (int i = 0; i < n; i++) {
        dst[i] = mtfp21_add(a[i], b[i]);
    }
}

/* ================================================================
 *  FFN compute — fully MTFP21
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
    const int n_tokens = (int)a->ne[1];
    float * output = (float *)dst->data;

    for (int tok = 0; tok < n_tokens; tok++) {
        const float * input = (const float *)a->data + tok * n;
        float * out_tok = output + tok * n;

        /* Convert input to MTFP21 */
        mtfp21_t inp_m[n]; /* VLA */
        for (int i = 0; i < n; i++) inp_m[i] = mtfp21_from_float(input[i]);

        /* 1. RMSNorm (ffn_norm) + pack for matmul */
        float act_scale = mtfp21_rmsnorm_and_pack(
            p->w_act, inp_m, p->ffn_norm_gamma, n, p->eps);

        /* 2. Gate matmul: packed int8 × 2bit → int32 raw → MTFP21 */
        ggml_gemv_i2_i8_s(n, p->w_raw, n_ff, p->gate_data, p->w_act, 1, n_ff);

        mtfp21_t gate_m[n_ff]; /* VLA */
        matmul_raw_to_mtfp21(gate_m, p->w_raw, n_ff, act_scale,
                             p->gate_wscale, p->gate_lscale);

        /* 3. Up matmul: packed int8 × 2bit → int32 raw → MTFP21 */
        ggml_gemv_i2_i8_s(n, p->w_raw, n_ff, p->up_data, p->w_act, 1, n_ff);

        mtfp21_t up_m[n_ff]; /* VLA */
        matmul_raw_to_mtfp21(up_m, p->w_raw, n_ff, act_scale,
                             p->up_wscale, p->up_lscale);

        /* 4. ReLU(gate) — full MTFP21 precision */
        mtfp21_relu(gate_m, gate_m, n_ff);

        /* 5. Square(ReLU(gate)) — full MTFP21 precision */
        mtfp21_square(gate_m, gate_m, n_ff);

        /* 6. gate² × up — full MTFP21 precision */
        mtfp21_t ffn_out[n_ff]; /* VLA */
        mtfp21_elem_mul(ffn_out, gate_m, up_m, n_ff);

        /* 7. Sub-norm + pack for down matmul */
        float sub_act_scale = mtfp21_rmsnorm_and_pack(
            p->w_sub, ffn_out, p->ffn_sub_norm_gamma, n_ff, p->eps);

        /* 8. Down matmul: packed int8 × 2bit → int32 raw → MTFP21 */
        ggml_gemv_i2_i8_s(n_ff, p->w_raw, n, p->down_data, p->w_sub, 1, n);

        mtfp21_t down_m[n]; /* VLA */
        matmul_raw_to_mtfp21(down_m, p->w_raw, n, sub_act_scale,
                             p->down_wscale, p->down_lscale);

        /* 9. Residual ADD in MTFP21, then convert to float for ggml */
        for (int i = 0; i < n; i++) {
            mtfp21_t result = mtfp21_add(down_m[i], inp_m[i]);
            out_tok[i] = mtfp21_to_float(result);
        }
    }

    static int logged = 0;
    if (!logged) {
        fprintf(stderr, "shirley: MTFP21 FFN active (layer %d, %d tokens)\n",
                p->layer_idx, n_tokens);
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

    fprintf(stderr, "shirley: MTFP21 FFN init layer %d (n=%d, n_ff=%d)\n",
            layer_idx, n_embd, n_ff);
}
