/*
 * shirley_attn.cpp — Shirley attention: adaptive-width MTFP, zero float
 *
 * QKV matmuls via MTFP16 sign_epi16 (no float conversion).
 * RoPE via precomputed CONST tables. Softmax via mtfp21_exp.
 * Q@K^T and attn@V as MTFP21 dot products.
 * KV cache stores MTFP21 values (as float for now — geometric cache later).
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
#include "shirley_attn.h"

/* ================================================================
 *  MTFP16 conversions (same as in shirley_ffn.cpp — should factor out)
 * ================================================================ */

#define MTFP16_MANT_MAX_ATTN 29524

typedef struct { int16_t mantissa; int8_t exponent; } mtfp16_attn_t;

static int8_t mtfp16_block_align_attn(
    int16_t * dst_mant, const mtfp21_t * src, int n
) {
    /* Convert + find max exponent */
    int8_t max_exp = -128;
    mtfp16_attn_t tmp[n]; /* VLA */
    for (int i = 0; i < n; i++) {
        int32_t m = src[i].mantissa;
        int exp = src[i].exponent;
        while (m > MTFP16_MANT_MAX_ATTN || m < -MTFP16_MANT_MAX_ATTN) {
            int32_t rem = m % 3; m /= 3;
            if (rem == 2) m++; else if (rem == -2) m--;
            exp++;
        }
        tmp[i].mantissa = (int16_t)m;
        tmp[i].exponent = (int8_t)exp;
        if (m != 0 && exp > max_exp) max_exp = exp;
    }
    if (max_exp == -128) max_exp = 0;

    for (int i = 0; i < n; i++) {
        if (tmp[i].mantissa == 0) { dst_mant[i] = 0; continue; }
        int shift = max_exp - tmp[i].exponent;
        int32_t m = (int32_t)tmp[i].mantissa;
        for (int s = 0; s < shift && s < 20; s++) {
            int32_t rem = m % 3; m /= 3;
            if (rem == 2) m++; else if (rem == -2) m--;
        }
        dst_mant[i] = (int16_t)m;
    }
    return max_exp;
}

/* ================================================================
 *  RoPE in MTFP21 — precomputed CONST tables
 * ================================================================ */

static void shirley_rope_mtfp21(
    mtfp21_t * dst, const mtfp21_t * src,
    const float * cos_table, const float * sin_table, int head_dim
) {
    int half = head_dim / 2;
    for (int i = 0; i < half; i++) {
        mtfp21_t cos_v = mtfp21_from_float(cos_table[i]);
        mtfp21_t sin_v = mtfp21_from_float(sin_table[i]);
        mtfp21_t x0 = src[i];
        mtfp21_t x1 = src[i + half];
        dst[i] = mtfp21_add(mtfp21_mul(x0, cos_v), mtfp21_neg(mtfp21_mul(x1, sin_v)));
        dst[i + half] = mtfp21_add(mtfp21_mul(x0, sin_v), mtfp21_mul(x1, cos_v));
    }
}

/* ================================================================
 *  MTFP21 softmax
 * ================================================================ */

static void shirley_softmax_mtfp21(mtfp21_t * dst, const mtfp21_t * src, int n) {
    mtfp21_t max_val = src[0];
    for (int i = 1; i < n; i++)
        if (mtfp21_cmp(src[i], max_val) > 0) max_val = src[i];

    mtfp21_t sum = {0, 0};
    for (int i = 0; i < n; i++) {
        dst[i] = mtfp21_exp(mtfp21_add(src[i], mtfp21_neg(max_val)));
        sum = mtfp21_add(sum, dst[i]);
    }

    mtfp21_t inv_sum = mtfp21_div(MTFP21_ONE_VAL, sum);
    for (int i = 0; i < n; i++)
        dst[i] = mtfp21_mul(dst[i], inv_sum);
}

/* ================================================================
 *  Attention compute — adaptive-width MTFP
 * ================================================================ */

extern "C"
void shirley_attn_compute(
    struct ggml_tensor * dst,
    const struct ggml_tensor * a,
    int ith, int nth,
    void * userdata
) {
    if (ith != 0) return;

    struct shirley_attn_params * p = (struct shirley_attn_params *)userdata;
    const int n = p->n_embd;
    const int n_head = p->n_head;
    const int n_kv = p->n_kv_head;
    const int hd = p->head_dim;
    const int n_tokens = (int)a->ne[1];
    float * output = (float *)dst->data;

    for (int tok = 0; tok < n_tokens; tok++) {
        const float * input = (const float *)a->data + tok * n;
        float * out_tok = output + tok * n;
        int pos = p->kv_pos + tok;

        /* Convert input to MTFP21 */
        mtfp21_t inp_m[n]; /* VLA */
        for (int i = 0; i < n; i++) inp_m[i] = mtfp21_from_float(input[i]);

        /* 0. attn_norm: MTFP21 RMSNorm */
        {
            mtfp21_t sum_sq = {0, 0};
            for (int i = 0; i < n; i++)
                sum_sq = mtfp21_add(sum_sq, mtfp21_mul(inp_m[i], inp_m[i]));
            mtfp21_t mean = mtfp21_div_scalar(sum_sq, n);
            mtfp21_t scale = mtfp21_rsqrt(mtfp21_add(mean, mtfp21_from_float(p->eps)));
            for (int i = 0; i < n; i++) {
                inp_m[i] = mtfp21_mul(inp_m[i], scale);
                if (p->attn_norm_gamma)
                    inp_m[i] = mtfp21_mul(inp_m[i], mtfp21_from_float(p->attn_norm_gamma[i]));
            }
        }

        /* 1. Block-align for QKV matmuls (MTFP21 → MTFP16) */
        int16_t act_mant[n]; /* VLA */
        int8_t block_exp = mtfp16_block_align_attn(act_mant, inp_m, n);

        /* 2. Q matmul: MTFP16 × ternary → MTFP21 (sign_epi16, zero float) */
        mtfp21_t q_m[n]; /* VLA */
        shirley_gemv_mtfp16(q_m, act_mant, block_exp, p->wq_data, n, n, p->wq_wscale);

        /* 3. K matmul */
        int kv_dim = n_kv * hd;
        mtfp21_t k_m[kv_dim]; /* VLA */
        shirley_gemv_mtfp16(k_m, act_mant, block_exp, p->wk_data, n, kv_dim, p->wk_wscale);

        /* 4. V matmul */
        mtfp21_t v_m[kv_dim]; /* VLA */
        shirley_gemv_mtfp16(v_m, act_mant, block_exp, p->wv_data, n, kv_dim, p->wv_wscale);

        /* 5. RoPE on Q and K */
        const float * cos_pos = p->rope_cos + pos * (hd / 2);
        const float * sin_pos = p->rope_sin + pos * (hd / 2);

        mtfp21_t q_rot[n]; /* VLA */
        for (int h = 0; h < n_head; h++)
            shirley_rope_mtfp21(q_rot + h * hd, q_m + h * hd, cos_pos, sin_pos, hd);

        mtfp21_t k_rot[kv_dim]; /* VLA */
        for (int h = 0; h < n_kv; h++)
            shirley_rope_mtfp21(k_rot + h * hd, k_m + h * hd, cos_pos, sin_pos, hd);

        /* 6. Store K and V in cache */
        for (int i = 0; i < kv_dim; i++) {
            p->k_cache[pos * kv_dim + i] = mtfp21_to_float(k_rot[i]);
            p->v_cache[pos * kv_dim + i] = mtfp21_to_float(v_m[i]);
        }

        /* 7. Attention: Q@K^T → softmax → attn@V */
        int gqa_ratio = n_head / n_kv;
        int kv_len = pos + 1;
        mtfp21_t attn_out[n]; /* VLA */

        for (int h = 0; h < n_head; h++) {
            int kv_h = h / gqa_ratio;
            mtfp21_t * qh = q_rot + h * hd;

            mtfp21_t scores[kv_len]; /* VLA */
            for (int t = 0; t < kv_len; t++) {
                mtfp21_t dot = {0, 0};
                for (int d = 0; d < hd; d++) {
                    mtfp21_t kval = mtfp21_from_float(p->k_cache[t * kv_dim + kv_h * hd + d]);
                    dot = mtfp21_add(dot, mtfp21_mul(qh[d], kval));
                }
                scores[t] = mtfp21_mul(dot, mtfp21_from_float(p->kq_scale));
            }

            shirley_softmax_mtfp21(scores, scores, kv_len);

            mtfp21_t * out_h = attn_out + h * hd;
            for (int d = 0; d < hd; d++) {
                mtfp21_t sum = {0, 0};
                for (int t = 0; t < kv_len; t++) {
                    mtfp21_t vval = mtfp21_from_float(p->v_cache[t * kv_dim + kv_h * hd + d]);
                    sum = mtfp21_add(sum, mtfp21_mul(scores[t], vval));
                }
                out_h[d] = sum;
            }
        }

        /* 8. attn_sub_norm + wo matmul (MTFP16, sign_epi16) */
        {
            mtfp21_t sum_sq = {0, 0};
            for (int i = 0; i < n; i++)
                sum_sq = mtfp21_add(sum_sq, mtfp21_mul(attn_out[i], attn_out[i]));
            mtfp21_t mean = mtfp21_div_scalar(sum_sq, n);
            mtfp21_t scale = mtfp21_rsqrt(mtfp21_add(mean, mtfp21_from_float(p->eps)));

            mtfp21_t normed[n]; /* VLA */
            for (int i = 0; i < n; i++) {
                normed[i] = mtfp21_mul(attn_out[i], scale);
                if (p->sub_norm_gamma)
                    normed[i] = mtfp21_mul(normed[i], mtfp21_from_float(p->sub_norm_gamma[i]));
            }

            /* wo matmul: MTFP16 × ternary → MTFP21 */
            int16_t wo_mant[n]; /* VLA */
            int8_t wo_exp = mtfp16_block_align_attn(wo_mant, normed, n);

            mtfp21_t wo_out[n]; /* VLA */
            shirley_gemv_mtfp16(wo_out, wo_mant, wo_exp, p->wo_data, n, n,
                p->wo_wscale * p->wo_lscale);

            /* 9. Residual ADD + convert to float */
            for (int i = 0; i < n; i++) {
                mtfp21_t residual = mtfp21_from_float(input[i]);
                out_tok[i] = mtfp21_to_float(mtfp21_add(wo_out[i], residual));
            }
        }
    }

    p->kv_pos += n_tokens;
    if (p->kv_pos > p->kv_len) p->kv_len = p->kv_pos;

    static int logged = 0;
    if (!logged) {
        fprintf(stderr, "shirley: MTFP16 attention active (layer %d, sign_epi16 QKV+wo, zero float matmul)\n",
                p->layer_idx);
        logged = 1;
    }
}

/* ================================================================
 *  Initialization
 * ================================================================ */

extern "C"
void shirley_attn_params_init(
    struct shirley_attn_params * p,
    int n_embd, int n_head, int n_kv_head, int head_dim,
    int max_seq_len, float eps, float kq_scale, int layer_idx,
    float rope_freq_base,
    const struct ggml_tensor * wq,
    const struct ggml_tensor * wk,
    const struct ggml_tensor * wv,
    const struct ggml_tensor * wo,
    const struct ggml_tensor * wo_scale_t,
    const struct ggml_tensor * attn_norm,
    const struct ggml_tensor * attn_sub_norm
) {
    p->n_embd = n_embd;
    p->n_head = n_head;
    p->n_kv_head = n_kv_head;
    p->head_dim = head_dim;
    p->n_rot = head_dim;
    p->eps = eps;
    p->kq_scale = kq_scale;
    p->layer_idx = layer_idx;
    p->rope_freq_base = rope_freq_base;
    p->max_seq_len = max_seq_len;

    p->wq_data = wq->data;
    p->wk_data = wk->data;
    p->wv_data = wv->data;
    p->wo_data = wo->data;

    p->wq_wscale = *(const float *)((const uint8_t *)wq->data + (wq->ne[0] * wq->ne[1] / 4));
    p->wk_wscale = *(const float *)((const uint8_t *)wk->data + (wk->ne[0] * wk->ne[1] / 4));
    p->wv_wscale = *(const float *)((const uint8_t *)wv->data + (wv->ne[0] * wv->ne[1] / 4));
    p->wo_wscale = *(const float *)((const uint8_t *)wo->data + (wo->ne[0] * wo->ne[1] / 4));
    p->wo_lscale = wo_scale_t ? *(const float *)wo_scale_t->data : 1.0f;
    p->attn_norm_gamma = attn_norm ? (const float *)attn_norm->data : NULL;
    p->sub_norm_gamma = attn_sub_norm ? (const float *)attn_sub_norm->data : NULL;

    /* RoPE sin/cos tables — CONST prime values */
    int half_dim = head_dim / 2;
    p->rope_cos = (float *)malloc(max_seq_len * half_dim * sizeof(float));
    p->rope_sin = (float *)malloc(max_seq_len * half_dim * sizeof(float));
    for (int pos = 0; pos < max_seq_len; pos++) {
        for (int i = 0; i < half_dim; i++) {
            float freq = 1.0f / powf(rope_freq_base, (float)(2 * i) / (float)head_dim);
            float angle = (float)pos * freq;
            p->rope_cos[pos * half_dim + i] = cosf(angle);
            p->rope_sin[pos * half_dim + i] = sinf(angle);
        }
    }

    /* KV cache */
    int kv_dim = n_kv_head * head_dim;
    p->k_cache = (float *)calloc(max_seq_len * kv_dim, sizeof(float));
    p->v_cache = (float *)calloc(max_seq_len * kv_dim, sizeof(float));
    p->kv_pos = 0;
    p->kv_len = 0;

    int max_dim = n_embd > (n_head * max_seq_len) ? n_embd : (n_head * max_seq_len);
    p->w_raw = (float *)malloc(max_dim * sizeof(float));

    p->ready = 1;

    fprintf(stderr, "shirley: MTFP16 attention init layer %d (sign_epi16 QKV+wo)\n", layer_idx);
}
