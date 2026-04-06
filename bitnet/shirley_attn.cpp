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
#include "shirley_mtfp16_ops.h"

#include "shirley_convert.h"

#include "3rdparty/llama.cpp/ggml/include/ggml.h"
#include "shirley_attn.h"
#include "shirley_profile.h"

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
    const int32_t * cos_mant, const int8_t * cos_exp,
    const int32_t * sin_mant, const int8_t * sin_exp,
    int head_dim
) {
    int half = head_dim / 2;
    for (int i = 0; i < half; i++) {
        mtfp21_t cos_v = {cos_mant[i], cos_exp[i]};
        mtfp21_t sin_v = {sin_mant[i], sin_exp[i]};
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

/* ================================================================
 *  Split-node attention: 5 callbacks, each a separate ggml graph node.
 *  ggml manages thread barriers. No spin-waits. Correct by construction.
 * ================================================================ */

extern "C"
void shirley_attn_prep(struct ggml_tensor * dst, const struct ggml_tensor * a, int ith, int nth, void * userdata) {
    if (ith != 0) return;
    struct shirley_attn_params * p = (struct shirley_attn_params *)userdata;
    const int n = p->n_embd;
    const int n_tokens = (int)a->ne[1];

    for (int tok = 0; tok < n_tokens; tok++) {
        p->mt_input = (const float *)a->data + tok * n;
        p->mt_output = (float *)dst->data + tok * n;
        p->mt_pos = p->kv_pos + tok;

        mtfp21_t inp_m[n]; /* VLA */
        for (int i = 0; i < n; i++) inp_m[i] = mtfp21_from_float(p->mt_input[i]);

        {
            int32_t m[n]; int8_t e[n]; /* VLA */
            for (int i = 0; i < n; i++) { m[i] = inp_m[i].mantissa; e[i] = inp_m[i].exponent; }
            mtfp21_rmsnorm_simd(m, e, m, e,
                p->attn_norm_gamma_mant, p->attn_norm_gamma_exp,
                n, p->eps_mant, p->eps_exp);
            for (int i = 0; i < n; i++) { inp_m[i].mantissa = m[i]; inp_m[i].exponent = e[i]; }
        }

        p->mt_bexp = mtfp16_block_align_attn(p->mt_act, inp_m, n);
    }
    /* Note: for multi-token, only the LAST token's state is in mt_*.
     * For single-token generation this is fine. Multi-token prompt eval
     * needs to be handled with a token loop in each node. For now,
     * this works correctly for n_tokens=1 (generation). */
}

extern "C"
void shirley_attn_qkv(struct ggml_tensor * dst, const struct ggml_tensor * a, int ith, int nth, void * userdata) {
    struct shirley_attn_params * p = (struct shirley_attn_params *)userdata;
    const int n = p->n_embd;
    const int kv_dim = p->n_kv_head * p->head_dim;
    mtfp21_t * mt_q = (mtfp21_t *)p->mt_qkv;
    mtfp21_t * mt_k = mt_q + n;
    mtfp21_t * mt_v = mt_k + kv_dim;

    shirley_gemv_mtfp16_part(mt_q, p->mt_act, p->mt_bexp, p->wq_data, n, n, p->wq_wscale, ith, nth);
    shirley_gemv_mtfp16_part(mt_k, p->mt_act, p->mt_bexp, p->wk_data, n, kv_dim, p->wk_wscale, ith, nth);
    shirley_gemv_mtfp16_part(mt_v, p->mt_act, p->mt_bexp, p->wv_data, n, kv_dim, p->wv_wscale, ith, nth);
}

extern "C"
void shirley_attn_body(struct ggml_tensor * dst, const struct ggml_tensor * a, int ith, int nth, void * userdata) {
    if (ith != 0) return;
    struct shirley_attn_params * p = (struct shirley_attn_params *)userdata;
    const int n = p->n_embd;
    const int n_head = p->n_head;
    const int n_kv = p->n_kv_head;
    const int hd = p->head_dim;
    const int kv_dim = n_kv * hd;
    int pos = p->mt_pos;
    mtfp21_t * mt_q = (mtfp21_t *)p->mt_qkv;
    mtfp21_t * mt_k = mt_q + n;
    mtfp21_t * mt_v = mt_k + kv_dim;

    /* RoPE */
    int half_d = hd / 2;
    const int32_t * cos_m = p->rope_cos_mant + pos * half_d;
    const int8_t  * cos_e = p->rope_cos_exp  + pos * half_d;
    const int32_t * sin_m = p->rope_sin_mant + pos * half_d;
    const int8_t  * sin_e = p->rope_sin_exp  + pos * half_d;

    mtfp21_t q_rot[n]; /* VLA */
    for (int h = 0; h < n_head; h++)
        shirley_rope_mtfp21(q_rot + h * hd, mt_q + h * hd, cos_m, cos_e, sin_m, sin_e, hd);
    mtfp21_t k_rot[kv_dim]; /* VLA */
    for (int h = 0; h < n_kv; h++)
        shirley_rope_mtfp21(k_rot + h * hd, mt_k + h * hd, cos_m, cos_e, sin_m, sin_e, hd);

    /* Cache store */
    for (int i = 0; i < kv_dim; i++) {
        int idx = pos * kv_dim + i;
        p->k_cache_mant[idx] = k_rot[i].mantissa;
        p->k_cache_exp[idx]  = k_rot[i].exponent;
        p->v_cache_mant[idx] = mt_v[i].mantissa;
        p->v_cache_exp[idx]  = mt_v[i].exponent;
    }

    /* Q@K^T → softmax → attn@V */
    int gqa_ratio = n_head / n_kv;
    int kv_len = pos + 1;
    mtfp21_t attn_out[n]; /* VLA */
    int32_t q_mant_arr[n]; int8_t q_exp_arr[n]; /* VLA */
    for (int i = 0; i < n; i++) { q_mant_arr[i] = q_rot[i].mantissa; q_exp_arr[i] = q_rot[i].exponent; }

    for (int h = 0; h < n_head; h++) {
        int kv_h = h / gqa_ratio;
        mtfp21_t scores[kv_len]; /* VLA */
        for (int t = 0; t < kv_len; t++) {
            mtfp21_t dot = mtfp21_dot_chunked(
                q_mant_arr + h * hd, q_exp_arr + h * hd,
                p->k_cache_mant + t * kv_dim + kv_h * hd,
                p->k_cache_exp  + t * kv_dim + kv_h * hd, hd);
            scores[t] = mtfp21_mul(dot, (mtfp21_t){p->kq_scale_mant, p->kq_scale_exp});
        }
        shirley_softmax_mtfp21(scores, scores, kv_len);

        int32_t score_mant[kv_len]; int8_t score_exp[kv_len]; /* VLA */
        for (int t = 0; t < kv_len; t++) { score_mant[t] = scores[t].mantissa; score_exp[t] = scores[t].exponent; }
        mtfp21_t * out_h = attn_out + h * hd;
        for (int d = 0; d < hd; d++) {
            int32_t v_d_m[kv_len]; int8_t v_d_e[kv_len]; /* VLA */
            for (int t = 0; t < kv_len; t++) {
                int vidx = t * kv_dim + kv_h * hd + d;
                v_d_m[t] = p->v_cache_mant[vidx]; v_d_e[t] = p->v_cache_exp[vidx];
            }
            out_h[d] = mtfp21_dot_chunked(score_mant, score_exp, v_d_m, v_d_e, kv_len);
        }
    }

    /* sub_norm → block-align for wo */
    {
        int32_t ao_m[n]; int8_t ao_e[n]; /* VLA */
        for (int i = 0; i < n; i++) { ao_m[i] = attn_out[i].mantissa; ao_e[i] = attn_out[i].exponent; }
        int32_t nm[n]; int8_t ne_arr[n]; /* VLA */
        mtfp21_rmsnorm_simd(nm, ne_arr, ao_m, ao_e,
            p->sub_norm_gamma_mant, p->sub_norm_gamma_exp,
            n, p->eps_mant, p->eps_exp);
        mtfp21_t normed[n]; /* VLA */
        for (int i = 0; i < n; i++) { normed[i].mantissa = nm[i]; normed[i].exponent = ne_arr[i]; }
        p->mt_wo_bexp = mtfp16_block_align_attn(p->mt_wo_act, normed, n);
    }
}

extern "C"
void shirley_attn_wo(struct ggml_tensor * dst, const struct ggml_tensor * a, int ith, int nth, void * userdata) {
    struct shirley_attn_params * p = (struct shirley_attn_params *)userdata;
    const int n = p->n_embd;
    mtfp21_t * mt_wo = (mtfp21_t *)p->mt_wo_out;
    shirley_gemv_mtfp16_part(mt_wo, p->mt_wo_act, p->mt_wo_bexp,
        p->wo_data, n, n, p->wo_wscale * p->wo_lscale, ith, nth);
}

extern "C"
void shirley_attn_finish(struct ggml_tensor * dst, const struct ggml_tensor * a, int ith, int nth, void * userdata) {
    if (ith != 0) return;
    struct shirley_attn_params * p = (struct shirley_attn_params *)userdata;
    const int n = p->n_embd;
    mtfp21_t * mt_wo = (mtfp21_t *)p->mt_wo_out;

    float * out = p->mt_output;
    const float * inp = p->mt_input;
    for (int i = 0; i < n; i++) {
        mtfp21_t residual = mtfp21_from_float(inp[i]);
        out[i] = mtfp21_to_float(mtfp21_add(mt_wo[i], residual));
    }

    p->kv_pos++;
    if (p->kv_pos > p->kv_len) p->kv_len = p->kv_pos;

    static int logged = 0;
    if (!logged) {
        fprintf(stderr, "shirley: MTFP16 attention active (split-node, threaded QKV+wo)\n");
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
    { mtfp21_t e = mtfp21_from_float(eps); p->eps_mant = e.mantissa; p->eps_exp = e.exponent; }
    p->kq_scale = kq_scale;
    { mtfp21_t kqs = mtfp21_from_float(kq_scale); p->kq_scale_mant = kqs.mantissa; p->kq_scale_exp = kqs.exponent; }
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
    shirley_convert_f32_to_mtfp21(
        &p->attn_norm_gamma_mant, &p->attn_norm_gamma_exp,
        attn_norm ? (const float *)attn_norm->data : NULL, n_embd);
    shirley_convert_f32_to_mtfp21(
        &p->sub_norm_gamma_mant, &p->sub_norm_gamma_exp,
        attn_sub_norm ? (const float *)attn_sub_norm->data : NULL, n_embd);

    /* RoPE sin/cos tables — precomputed as MTFP21 (CONST prime).
     * Transcendentals consumed at load time. Zero float at runtime. */
    int half_dim = head_dim / 2;
    int64_t rope_n = (int64_t)max_seq_len * half_dim;
    p->rope_cos_mant = (int32_t *)malloc(rope_n * sizeof(int32_t));
    p->rope_cos_exp  = (int8_t  *)malloc(rope_n * sizeof(int8_t));
    p->rope_sin_mant = (int32_t *)malloc(rope_n * sizeof(int32_t));
    p->rope_sin_exp  = (int8_t  *)malloc(rope_n * sizeof(int8_t));
    for (int pos = 0; pos < max_seq_len; pos++) {
        for (int i = 0; i < half_dim; i++) {
            float freq = 1.0f / powf(rope_freq_base, (float)(2 * i) / (float)head_dim);
            float angle = (float)pos * freq;
            mtfp21_t c = mtfp21_from_float(cosf(angle));
            mtfp21_t s = mtfp21_from_float(sinf(angle));
            int64_t idx = (int64_t)pos * half_dim + i;
            p->rope_cos_mant[idx] = c.mantissa;
            p->rope_cos_exp[idx]  = c.exponent;
            p->rope_sin_mant[idx] = s.mantissa;
            p->rope_sin_exp[idx]  = s.exponent;
        }
    }

    /* KV cache — native MTFP21 per element */
    int kv_dim = n_kv_head * head_dim;
    int64_t cache_n = (int64_t)max_seq_len * kv_dim;
    p->k_cache_mant = (int32_t *)calloc(cache_n, sizeof(int32_t));
    p->k_cache_exp  = (int8_t  *)calloc(cache_n, sizeof(int8_t));
    p->v_cache_mant = (int32_t *)calloc(cache_n, sizeof(int32_t));
    p->v_cache_exp  = (int8_t  *)calloc(cache_n, sizeof(int8_t));
    p->kv_pos = 0;
    p->kv_len = 0;

    int max_dim = n_embd > (n_head * max_seq_len) ? n_embd : (n_head * max_seq_len);
    p->w_raw = (float *)malloc(max_dim * sizeof(float));

    /* Split-node shared workspace */
    int kv_dim2 = n_kv_head * head_dim;
    p->mt_act    = (int16_t *)malloc(n_embd * sizeof(int16_t));
    p->mt_qkv    = malloc((n_embd + kv_dim2 + kv_dim2) * sizeof(mtfp21_t));
    p->mt_wo_act = (int16_t *)malloc(n_embd * sizeof(int16_t));
    p->mt_wo_out = malloc(n_embd * sizeof(mtfp21_t));

    p->ready = 1;

    fprintf(stderr, "shirley: MTFP16 attention init layer %d (sign_epi16 QKV+wo, threaded)\n", layer_idx);
}
