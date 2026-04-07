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

extern "C" {
void ggml_gemv_i2_i8_s(int n, float * s, size_t bs,
    const void * vx, const void * vy, int nr, int nc);
}
#include "shirley_profile.h"
#include "shirley_barrier.h"

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

extern "C"
void shirley_attn_compute(
    struct ggml_tensor * dst,
    const struct ggml_tensor * a,
    int ith, int nth,
    void * userdata
) {
    struct shirley_attn_params * p = (struct shirley_attn_params *)userdata;
    const int n = p->n_embd;
    const int n_head = p->n_head;
    const int n_kv = p->n_kv_head;
    const int hd = p->head_dim;
    const int kv_dim = n_kv * hd;
    const int n_tokens = (int)a->ne[1];
    float * output = (float *)dst->data;

    /* Phase managed per-token: 0 → 1 (norm) → 3 (body) → 0 (done) */

    for (int tok = 0; tok < n_tokens; tok++) {
        const float * input = (const float *)a->data + tok * n;
        float * out_tok = output + tok * n;
        int pos = p->kv_pos + tok;

        SP_START;

        /* ---- attn_norm (thread 0) ---- */
        if (ith == 0) {
            shirley_rmsnorm_quantize(p->mt_act_i8, input, p->attn_norm_gamma_f32, n, p->eps, 80);
            SP_LAP(attn_norm);
            __atomic_store_n(&p->mt_phase, 1, __ATOMIC_RELEASE);
        } else {
            while (__atomic_load_n(&p->mt_phase, __ATOMIC_ACQUIRE) < 1) { _mm_pause(); }
        }

        /* ---- PHASE 1: QKV matmul — ALL threads, partitioned by rows ---- */
        {
            /* Q matmul: n_embd output rows */
            int rows_per = (n + nth - 1) / nth;
            int r0 = ith * rows_per;
            int r1 = r0 + rows_per; if (r1 > n) r1 = n;
            if (r0 < r1) {
                float q_tmp[r1 - r0]; /* VLA */
                ggml_gemv_i2_i8_s(n, q_tmp, r1 - r0,
                    (const char *)p->wq_data + (int64_t)r0 * (n / 4),
                    p->mt_act_i8, 1, r1 - r0);
                for (int i = 0; i < r1 - r0; i++)
                    p->mt_q_f[r0 + i] = q_tmp[i] * p->wq_wscale;
            }

            /* K matmul: kv_dim output rows */
            rows_per = (kv_dim + nth - 1) / nth;
            r0 = ith * rows_per;
            r1 = r0 + rows_per; if (r1 > kv_dim) r1 = kv_dim;
            if (r0 < r1) {
                float k_tmp[r1 - r0]; /* VLA */
                ggml_gemv_i2_i8_s(n, k_tmp, r1 - r0,
                    (const char *)p->wk_data + (int64_t)r0 * (n / 4),
                    p->mt_act_i8, 1, r1 - r0);
                for (int i = 0; i < r1 - r0; i++)
                    p->mt_k_f[r0 + i] = k_tmp[i] * p->wk_wscale;
            }

            /* V matmul: kv_dim output rows */
            r0 = ith * ((kv_dim + nth - 1) / nth);
            r1 = r0 + ((kv_dim + nth - 1) / nth); if (r1 > kv_dim) r1 = kv_dim;
            if (r0 < r1) {
                float v_tmp[r1 - r0]; /* VLA */
                ggml_gemv_i2_i8_s(n, v_tmp, r1 - r0,
                    (const char *)p->wv_data + (int64_t)r0 * (n / 4),
                    p->mt_act_i8, 1, r1 - r0);
                for (int i = 0; i < r1 - r0; i++)
                    p->mt_v_f[r0 + i] = v_tmp[i] * p->wv_wscale;
            }
        }

        /* Barrier: wait for all threads to finish QKV */
        __atomic_fetch_add(&p->mt_threads_done, 1, __ATOMIC_ACQ_REL);
        if (ith == 0) {
            while (__atomic_load_n(&p->mt_threads_done, __ATOMIC_ACQUIRE) < nth) { _mm_pause(); }
            __atomic_store_n(&p->mt_threads_done, 0, __ATOMIC_RELEASE);
            SP_LAP(attn_qkv_matmul);

            /* ---- PHASE 2: RoPE + KV cache (thread 0) ---- */
            float * q_f = p->mt_q_f;
            float * k_f = p->mt_k_f;
            float * v_f = p->mt_v_f;

            int half_d = hd / 2;
            float * rope_cos_f = (float *)__builtin_alloca(half_d * sizeof(float));
            float * rope_sin_f = (float *)__builtin_alloca(half_d * sizeof(float));
            for (int i = 0; i < half_d; i++) {
                rope_cos_f[i] = mtfp21_to_float((mtfp21_t){p->rope_cos_mant[pos * half_d + i], p->rope_cos_exp[pos * half_d + i]});
                rope_sin_f[i] = mtfp21_to_float((mtfp21_t){p->rope_sin_mant[pos * half_d + i], p->rope_sin_exp[pos * half_d + i]});
            }
            for (int h = 0; h < n_head; h++) {
                float * qh = q_f + h * hd;
                for (int i = 0; i < half_d; i++) {
                    float x0 = qh[i], x1 = qh[i + half_d];
                    qh[i]          = x0 * rope_cos_f[i] - x1 * rope_sin_f[i];
                    qh[i + half_d] = x0 * rope_sin_f[i] + x1 * rope_cos_f[i];
                }
            }
            for (int h = 0; h < n_kv; h++) {
                float * kh = k_f + h * hd;
                for (int i = 0; i < half_d; i++) {
                    float x0 = kh[i], x1 = kh[i + half_d];
                    kh[i]          = x0 * rope_cos_f[i] - x1 * rope_sin_f[i];
                    kh[i + half_d] = x0 * rope_sin_f[i] + x1 * rope_cos_f[i];
                }
            }
            SP_LAP(attn_rope);

            float * k_cache_f = (float *)p->k_cache_mant;
            float * v_cache_f = (float *)p->v_cache_mant;
            memcpy(k_cache_f + pos * kv_dim, k_f, kv_dim * sizeof(float));
            memcpy(v_cache_f + pos * kv_dim, v_f, kv_dim * sizeof(float));
            SP_LAP(attn_kv_cache);

            /* Q@K^T + softmax + attn@V — thread 0 only */
            int gqa_ratio = n_head / n_kv;
            int kv_len = pos + 1;

            for (int h = 0; h < n_head; h++) {
                int kv_h = h / gqa_ratio;
                float * qh = q_f + h * hd;

                float scores[kv_len]; /* VLA */
                for (int t = 0; t < kv_len; t++) {
                    float dot = 0.0f;
                    float * kt = k_cache_f + t * kv_dim + kv_h * hd;
                    for (int d = 0; d < hd; d++) dot += qh[d] * kt[d];
                    scores[t] = dot * p->kq_scale;
                }

                {
                    float max_s = scores[0];
                    for (int t = 1; t < kv_len; t++) if (scores[t] > max_s) max_s = scores[t];
                    float sum = 0.0f;
                    for (int t = 0; t < kv_len; t++) { scores[t] = expf(scores[t] - max_s); sum += scores[t]; }
                    float inv_sum = 1.0f / sum;
                    for (int t = 0; t < kv_len; t++) scores[t] *= inv_sum;
                }

                float * out_h = p->mt_attn_out + h * hd;
                for (int d = 0; d < hd; d++) {
                    float sum = 0.0f;
                    for (int t = 0; t < kv_len; t++) {
                        sum += scores[t] * v_cache_f[t * kv_dim + kv_h * hd + d];
                    }
                    out_h[d] = sum;
                }
            }
            SP_LAP(attn_av);

            shirley_rmsnorm_quantize(p->mt_sub_i8, p->mt_attn_out, p->sub_norm_gamma_f32, n, p->eps, 80);
            SP_LAP(attn_sub_norm);

            __atomic_store_n(&p->mt_phase, 3, __ATOMIC_RELEASE);
        } else {
            while (__atomic_load_n(&p->mt_phase, __ATOMIC_ACQUIRE) < 3) { _mm_pause(); }
        }

        /* ---- wo matmul — ALL threads, partitioned by rows ---- */
        {
            int rows_per = (n + nth - 1) / nth;
            int r0 = ith * rows_per;
            int r1 = r0 + rows_per; if (r1 > n) r1 = n;
            if (r0 < r1) {
                float wo_tmp[r1 - r0]; /* VLA */
                ggml_gemv_i2_i8_s(n, wo_tmp, r1 - r0,
                    (const char *)p->wo_data + (int64_t)r0 * (n / 4),
                    p->mt_sub_i8, 1, r1 - r0);
                float wo_cs = p->wo_wscale * p->wo_lscale;
                for (int i = 0; i < r1 - r0; i++) {
                    p->mt_wo_f[r0 + i] = wo_tmp[i] * wo_cs;
                }
            }
        }

        /* Barrier: all threads finish wo */
        __atomic_fetch_add(&p->mt_threads_done, 1, __ATOMIC_ACQ_REL);
        if (ith == 0) {
            while (__atomic_load_n(&p->mt_threads_done, __ATOMIC_ACQUIRE) < nth) { _mm_pause(); }
            __atomic_store_n(&p->mt_threads_done, 0, __ATOMIC_RELEASE);
            SP_LAP(attn_wo);

            for (int i = 0; i < n; i++) {
                out_tok[i] = p->mt_wo_f[i] + input[i];
            }
            SP_LAP(attn_residual);
            SP_TOKEN();

            __atomic_store_n(&p->mt_phase, 0, __ATOMIC_RELEASE);
        } else {
            while (__atomic_load_n(&p->mt_phase, __ATOMIC_ACQUIRE) != 0) { _mm_pause(); }
        }
    }

    if (ith == 0) {
        p->kv_pos += n_tokens;
        if (p->kv_pos > p->kv_len) p->kv_len = p->kv_pos;
        static int logged = 0;
        if (!logged) {
            fprintf(stderr, "shirley: multi-threaded attention active (sign_epi8 QKV+wo, float body)\n");
            logged = 1;
        }
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
    /* Float gammas for the kernel-based path */
    p->attn_norm_gamma_f32 = attn_norm ? (const float *)attn_norm->data : NULL;
    /* Sub norm gamma: need float pointer for shirley_rmsnorm_quantize */
    /* Stored as a field on the params struct. The float data lives in the model tensor. */
    /* For now, reuse attn_sub_norm's float data. Note: sub_norm gamma is [n_embd], not [n_ff]. */
    if (attn_sub_norm) {
        const float * sg = (const float *)attn_sub_norm->data;
        p->sub_norm_gamma_q14 = (int16_t *)malloc(n_embd * sizeof(int16_t));
        for (int i = 0; i < n_embd; i++)
            p->sub_norm_gamma_q14[i] = (int16_t)(sg[i] * 16384.0f + 0.5f);
        p->sub_norm_gamma_f32 = sg;
    } else {
        p->sub_norm_gamma_q14 = NULL;
        p->sub_norm_gamma_f32 = NULL;
    }

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

    /* Threading workspace */
    p->mt_phase = 0;
    p->mt_threads_done = 0;
    int kv_dim2 = n_kv_head * head_dim;
    p->mt_act_i8   = (int8_t *)malloc(n_embd * sizeof(int8_t));
    p->mt_q_f      = (float *)malloc(n_embd * sizeof(float));
    p->mt_k_f      = (float *)malloc(kv_dim2 * sizeof(float));
    p->mt_v_f      = (float *)malloc(kv_dim2 * sizeof(float));
    p->mt_attn_out = (float *)malloc(n_embd * sizeof(float));
    p->mt_sub_i8   = (int8_t *)malloc(n_embd * sizeof(int8_t));
    p->mt_wo_f     = (float *)malloc(n_embd * sizeof(float));

    p->ready = 1;

    fprintf(stderr, "shirley: MTFP16 attention init layer %d (sign_epi16 QKV+wo, threaded)\n", layer_idx);
}
