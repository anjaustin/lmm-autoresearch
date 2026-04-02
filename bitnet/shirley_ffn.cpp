/*
 * shirley_ffn.cpp — Shirley FFN: adaptive-width MTFP, zero float
 *
 * Adaptive-width MTFP:
 *   MTFP21 (int32 mantissa) — RMSNorm rsqrt, precision scalar ops
 *   MTFP16 (int16 mantissa) — matmul activations via sign_epi16
 *
 * No float conversion in the matmul path.
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#define restrict
#include "shirley_kernels.h"
#include "shirley_mtfp16_matmul.h"

#include "3rdparty/llama.cpp/ggml/include/ggml.h"
#include "shirley_ffn.h"

/* ================================================================
 *  MTFP16 type and conversions (inline for this file)
 * ================================================================ */

#define MTFP16_MANT_MAX 29524  /* (3^10 - 1) / 2 */

typedef struct { int16_t mantissa; int8_t exponent; } mtfp16_local_t;

static mtfp16_local_t to_mtfp16(mtfp21_t a) {
    mtfp16_local_t r;
    if (a.mantissa == 0) { r.mantissa = 0; r.exponent = 0; return r; }
    int32_t m = a.mantissa;
    int exp = a.exponent;
    while (m > MTFP16_MANT_MAX || m < -MTFP16_MANT_MAX) {
        int32_t rem = m % 3;
        m /= 3;
        if (rem == 2) m++;
        else if (rem == -2) m--;
        exp++;
    }
    r.mantissa = (int16_t)m;
    r.exponent = (int8_t)exp;
    return r;
}

/* Block-align MTFP16 vector: shared exponent, int16 mantissas for SIMD */
static int8_t mtfp16_block_align(
    int16_t * dst_mant, const mtfp21_t * src, int n
) {
    /* Convert to MTFP16 first */
    mtfp16_local_t tmp[n]; /* VLA */
    for (int i = 0; i < n; i++) tmp[i] = to_mtfp16(src[i]);

    /* Find max exponent */
    int8_t max_exp = -128;
    for (int i = 0; i < n; i++) {
        if (tmp[i].mantissa != 0 && tmp[i].exponent > max_exp)
            max_exp = tmp[i].exponent;
    }
    if (max_exp == -128) max_exp = 0;

    /* Align mantissas to max_exp */
    for (int i = 0; i < n; i++) {
        if (tmp[i].mantissa == 0) { dst_mant[i] = 0; continue; }
        int shift = max_exp - tmp[i].exponent;
        int32_t m = (int32_t)tmp[i].mantissa;
        for (int s = 0; s < shift && s < 20; s++) {
            int32_t rem = m % 3;
            m /= 3;
            if (rem == 2) m++;
            else if (rem == -2) m--;
        }
        dst_mant[i] = (int16_t)m;
    }
    return max_exp;
}

/* ================================================================
 *  MTFP21 RMSNorm → block-aligned MTFP16 for matmul
 * ================================================================ */
static int8_t mtfp21_rmsnorm_to_mtfp16(
    int16_t * dst_mant,
    const mtfp21_t * src,
    const float * gamma,
    int n, float eps
) {
    mtfp21_t sum_sq = {0, 0};
    for (int i = 0; i < n; i++)
        sum_sq = mtfp21_add(sum_sq, mtfp21_mul(src[i], src[i]));
    mtfp21_t mean = mtfp21_div_scalar(sum_sq, n);
    mtfp21_t scale = mtfp21_rsqrt(mtfp21_add(mean, mtfp21_from_float(eps)));

    mtfp21_t normed[n]; /* VLA */
    for (int i = 0; i < n; i++) {
        normed[i] = mtfp21_mul(src[i], scale);
        if (gamma)
            normed[i] = mtfp21_mul(normed[i], mtfp21_from_float(gamma[i]));
    }

    return mtfp16_block_align(dst_mant, normed, n);
}

/* ================================================================
 *  MTFP21 operations (unchanged — full precision between matmuls)
 * ================================================================ */
static void mtfp21_relu(mtfp21_t * dst, const mtfp21_t * src, int n) {
    for (int i = 0; i < n; i++)
        dst[i] = (src[i].mantissa > 0) ? src[i] : (mtfp21_t){0, 0};
}

static void mtfp21_square(mtfp21_t * dst, const mtfp21_t * src, int n) {
    for (int i = 0; i < n; i++)
        dst[i] = mtfp21_mul(src[i], src[i]);
}

static void mtfp21_elem_mul(mtfp21_t * dst, const mtfp21_t * a, const mtfp21_t * b, int n) {
    for (int i = 0; i < n; i++)
        dst[i] = mtfp21_mul(a[i], b[i]);
}

/* ================================================================
 *  FFN compute — adaptive-width MTFP, zero float in matmul path
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

        /* 1. RMSNorm → block-aligned MTFP16 for matmul (MTFP21 precision norm) */
        int16_t act_mant[n]; /* VLA — block-aligned mantissas */
        int8_t block_exp = mtfp21_rmsnorm_to_mtfp16(
            act_mant, inp_m, p->ffn_norm_gamma, n, p->eps);

        /* 2. Gate matmul: MTFP16 × ternary → MTFP21 (pure integer, sign_epi16) */
        mtfp21_t gate_m[n_ff]; /* VLA */
        shirley_gemv_mtfp16(gate_m, act_mant, block_exp,
            p->gate_data, n, n_ff, p->gate_wscale * p->gate_lscale);

        /* 3. Up matmul: same path */
        mtfp21_t up_m[n_ff]; /* VLA */
        shirley_gemv_mtfp16(up_m, act_mant, block_exp,
            p->up_data, n, n_ff, p->up_wscale * p->up_lscale);

        /* 4-6. ReLU → Square → gate² × up (full MTFP21 precision) */
        mtfp21_relu(gate_m, gate_m, n_ff);
        mtfp21_square(gate_m, gate_m, n_ff);
        mtfp21_t ffn_out[n_ff]; /* VLA */
        mtfp21_elem_mul(ffn_out, gate_m, up_m, n_ff);

        /* 7. Sub-norm → block-aligned MTFP16 for down matmul */
        int16_t sub_mant[n_ff]; /* VLA */
        int8_t sub_exp = mtfp21_rmsnorm_to_mtfp16(
            sub_mant, ffn_out, p->ffn_sub_norm_gamma, n_ff, p->eps);

        /* 8. Down matmul: MTFP16 × ternary → MTFP21 */
        mtfp21_t down_m[n]; /* VLA */
        shirley_gemv_mtfp16(down_m, sub_mant, sub_exp,
            p->down_data, n_ff, n, p->down_wscale * p->down_lscale);

        /* 9. Residual ADD in MTFP21, convert to float for ggml */
        for (int i = 0; i < n; i++) {
            mtfp21_t result = mtfp21_add(down_m[i], inp_m[i]);
            out_tok[i] = mtfp21_to_float(result);
        }
    }

    static int logged = 0;
    if (!logged) {
        fprintf(stderr, "shirley: MTFP16 FFN active (layer %d, sign_epi16 matmul, zero float)\n",
                p->layer_idx);
        logged = 1;
    }
}

/* ================================================================
 *  Initialization (unchanged)
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

    fprintf(stderr, "shirley: MTFP16 FFN init layer %d (n=%d, n_ff=%d)\n",
            layer_idx, n_embd, n_ff);
}
