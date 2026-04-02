/*
 * shirley_attn.cpp — Shirley attention: fully MTFP21
 *
 * QKV matmuls → RoPE (CONST tables) → Q@K^T → softmax (EXP prime) → attn@V
 * All values MTFP21. Int8 only at ternary matmul wire.
 * RoPE sin/cos are precomputed CONST prime values — no runtime transcendentals.
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#define restrict
#include "shirley_kernels.h"

#include "3rdparty/llama.cpp/ggml/include/ggml.h"
#include "shirley_attn.h"

extern "C" {
void ggml_gemv_i2_i8_s(int n, float * s, size_t bs,
    const void * vx, const void * vy, int nr, int nc);
}

/* ================================================================
 *  MTFP21 helpers (shared with shirley_ffn.cpp — should factor out)
 * ================================================================ */

static float mtfp21_pack_for_matmul_attn(
    int8_t * dst_i8, const mtfp21_t * src, int n, int range
) {
    float max_abs = 0.0f;
    for (int i = 0; i < n; i++) {
        float f = mtfp21_to_float(src[i]);
        float a = f > 0 ? f : -f;
        if (a > max_abs) max_abs = a;
    }
    if (max_abs < 1e-30f) { memset(dst_i8, 0, n); return 0.0f; }
    float scale = (float)range / max_abs;
    for (int i = 0; i < n; i++) {
        float f = mtfp21_to_float(src[i]);
        int32_t r = (int32_t)(f * scale + (f >= 0 ? 0.5f : -0.5f));
        if (r > range) r = range;
        if (r < -range) r = -range;
        dst_i8[i] = (int8_t)r;
    }
    return scale;
}

static void matmul_raw_to_mtfp21_attn(
    mtfp21_t * dst, const float * raw, int n,
    float act_scale, float weight_scale
) {
    float combined_f = weight_scale / act_scale;
    mtfp21_t combined = mtfp21_from_float(combined_f);
    for (int i = 0; i < n; i++) {
        dst[i] = mtfp21_mul(mtfp21_from_float(raw[i]), combined);
    }
}

/* ================================================================
 *  RoPE in MTFP21 — using precomputed CONST tables
 *
 *  For each pair (x[2i], x[2i+1]) at position pos:
 *    x_rot[2i]   = x[2i] * cos[pos][i] - x[2i+1] * sin[pos][i]
 *    x_rot[2i+1] = x[2i] * sin[pos][i] + x[2i+1] * cos[pos][i]
 *
 *  cos/sin are CONST prime values — frozen at model load.
 *  The operation is MUL + ADD — two CPU-native primes.
 * ================================================================ */

static void shirley_rope_mtfp21(
    mtfp21_t * dst,         /* output: rotated [head_dim] */
    const mtfp21_t * src,   /* input: [head_dim] */
    const float * cos_table, /* precomputed cos[head_dim/2] for this position */
    const float * sin_table, /* precomputed sin[head_dim/2] for this position */
    int head_dim
) {
    int half = head_dim / 2;
    for (int i = 0; i < half; i++) {
        mtfp21_t cos_v = mtfp21_from_float(cos_table[i]);
        mtfp21_t sin_v = mtfp21_from_float(sin_table[i]);
        mtfp21_t x0 = src[i];
        mtfp21_t x1 = src[i + half];

        /* x_rot[i]      = x0 * cos - x1 * sin */
        dst[i] = mtfp21_add(
            mtfp21_mul(x0, cos_v),
            mtfp21_neg(mtfp21_mul(x1, sin_v))
        );
        /* x_rot[i+half] = x0 * sin + x1 * cos */
        dst[i + half] = mtfp21_add(
            mtfp21_mul(x0, sin_v),
            mtfp21_mul(x1, cos_v)
        );
    }
}

/* ================================================================
 *  MTFP21 matmul: dense × dense (for Q@K^T and attn@V)
 *
 *  C[i][j] = Σ_k A[i][k] * B[k][j]
 *  All in MTFP21. No SIMD optimization yet — correctness first.
 * ================================================================ */

static void shirley_matmul_mtfp21(
    mtfp21_t * dst,          /* [m × n] output */
    const mtfp21_t * a,      /* [m × k] */
    const mtfp21_t * b,      /* [k × n] */
    int m, int k, int n_cols
) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n_cols; j++) {
            mtfp21_t sum = {0, 0};
            for (int kk = 0; kk < k; kk++) {
                sum = mtfp21_add(sum, mtfp21_mul(a[i * k + kk], b[kk * n_cols + j]));
            }
            dst[i * n_cols + j] = sum;
        }
    }
}

/* ================================================================
 *  MTFP21 softmax (using the built mtfp21_exp)
 * ================================================================ */

static void shirley_softmax_mtfp21(
    mtfp21_t * dst,
    const mtfp21_t * src,
    int n
) {
    /* Find max */
    mtfp21_t max_val = src[0];
    for (int i = 1; i < n; i++) {
        if (mtfp21_cmp(src[i], max_val) > 0) max_val = src[i];
    }

    /* exp(x - max) and sum */
    mtfp21_t sum = {0, 0};
    for (int i = 0; i < n; i++) {
        dst[i] = mtfp21_exp(mtfp21_add(src[i], mtfp21_neg(max_val)));
        sum = mtfp21_add(sum, dst[i]);
    }

    /* Normalize */
    mtfp21_t inv_sum = mtfp21_div(MTFP21_ONE_VAL, sum);
    for (int i = 0; i < n; i++) {
        dst[i] = mtfp21_mul(dst[i], inv_sum);
    }
}

/* ================================================================
 *  Attention compute — fully MTFP21
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

    /* Process each token */
    for (int tok = 0; tok < n_tokens; tok++) {
        const float * input = (const float *)a->data + tok * n;
        float * out_tok = output + tok * n;
        int pos = p->kv_pos + tok;  /* position for RoPE */

        /* Convert input to MTFP21 */
        mtfp21_t inp_m[n]; /* VLA */
        for (int i = 0; i < n; i++) inp_m[i] = mtfp21_from_float(input[i]);

        /* 1. Q matmul: MTFP21 → pack → ternary matmul → MTFP21
         *    wq: [n_embd, n_embd] (n_head * head_dim = 2560 output) */
        int8_t act_i8[n]; /* VLA */
        float act_scale = mtfp21_pack_for_matmul_attn(act_i8, inp_m, n, 80);

        float q_raw[n]; /* n_embd = n_head * head_dim */
        ggml_gemv_i2_i8_s(n, q_raw, n, p->wq_data, act_i8, 1, n);

        mtfp21_t q_m[n]; /* [n_head * head_dim] */
        matmul_raw_to_mtfp21_attn(q_m, q_raw, n, act_scale, p->wq_wscale);

        /* 2. K matmul: output is [n_kv_head * head_dim = 640] */
        int kv_dim = n_kv * hd; /* 5 * 128 = 640 */
        float k_raw[kv_dim]; /* VLA */
        ggml_gemv_i2_i8_s(n, k_raw, kv_dim, p->wk_data, act_i8, 1, kv_dim);

        mtfp21_t k_m[kv_dim]; /* [n_kv_head * head_dim] */
        matmul_raw_to_mtfp21_attn(k_m, k_raw, kv_dim, act_scale, p->wk_wscale);

        /* 3. V matmul: same shape as K */
        float v_raw[kv_dim]; /* VLA */
        ggml_gemv_i2_i8_s(n, v_raw, kv_dim, p->wv_data, act_i8, 1, kv_dim);

        mtfp21_t v_m[kv_dim]; /* [n_kv_head * head_dim] */
        matmul_raw_to_mtfp21_attn(v_m, v_raw, kv_dim, act_scale, p->wv_wscale);

        /* 4. RoPE on Q (per head) and K (per kv_head) */
        const float * cos_pos = p->rope_cos + pos * (hd / 2);
        const float * sin_pos = p->rope_sin + pos * (hd / 2);

        mtfp21_t q_rot[n]; /* VLA */
        for (int h = 0; h < n_head; h++) {
            shirley_rope_mtfp21(q_rot + h * hd, q_m + h * hd,
                                cos_pos, sin_pos, hd);
        }

        mtfp21_t k_rot[kv_dim]; /* VLA */
        for (int h = 0; h < n_kv; h++) {
            shirley_rope_mtfp21(k_rot + h * hd, k_m + h * hd,
                                cos_pos, sin_pos, hd);
        }

        /* 5. Store K and V in cache (as float for now) */
        for (int i = 0; i < kv_dim; i++) {
            p->k_cache[pos * kv_dim + i] = mtfp21_to_float(k_rot[i]);
            p->v_cache[pos * kv_dim + i] = mtfp21_to_float(v_m[i]);
        }

        /* 6. Attention: per head, Q@K^T → softmax → attn@V
         *    GQA: n_head=20, n_kv_head=5, so 4 Q heads per KV head */
        int gqa_ratio = n_head / n_kv;
        int kv_len = pos + 1; /* total keys available (including this token) */

        mtfp21_t attn_out[n]; /* VLA: [n_head * head_dim] */

        for (int h = 0; h < n_head; h++) {
            int kv_h = h / gqa_ratio; /* which KV head this Q head attends to */

            /* Q for this head: q_rot[h*hd .. h*hd+hd-1] */
            mtfp21_t * qh = q_rot + h * hd;

            /* Compute attention scores: Q @ K^T for all cached positions */
            mtfp21_t scores[kv_len]; /* VLA */
            for (int t = 0; t < kv_len; t++) {
                /* K at position t, head kv_h */
                mtfp21_t dot = {0, 0};
                for (int d = 0; d < hd; d++) {
                    mtfp21_t kval = mtfp21_from_float(
                        p->k_cache[t * kv_dim + kv_h * hd + d]);
                    dot = mtfp21_add(dot, mtfp21_mul(qh[d], kval));
                }
                /* Scale by 1/sqrt(head_dim) */
                scores[t] = mtfp21_mul(dot, mtfp21_from_float(p->kq_scale));
            }

            /* Causal softmax (all positions up to current are visible) */
            shirley_softmax_mtfp21(scores, scores, kv_len);

            /* Weighted sum of V: Σ scores[t] * V[t] */
            mtfp21_t * out_h = attn_out + h * hd;
            for (int d = 0; d < hd; d++) {
                mtfp21_t sum = {0, 0};
                for (int t = 0; t < kv_len; t++) {
                    mtfp21_t vval = mtfp21_from_float(
                        p->v_cache[t * kv_dim + kv_h * hd + d]);
                    sum = mtfp21_add(sum, mtfp21_mul(scores[t], vval));
                }
                out_h[d] = sum;
            }
        }

        /* 7. Convert MTFP21 attention output to float for ggml */
        for (int i = 0; i < n; i++) {
            out_tok[i] = mtfp21_to_float(attn_out[i]);
        }
    }

    /* Update cache position */
    p->kv_pos += n_tokens;
    if (p->kv_pos > p->kv_len) p->kv_len = p->kv_pos;

    static int logged = 0;
    if (!logged) {
        fprintf(stderr, "shirley: MTFP21 attention active (layer %d, %d tokens, kv_pos=%d)\n",
                p->layer_idx, n_tokens, p->kv_pos);
        logged = 1;
    }
}

/* ================================================================
 *  Initialization: precompute RoPE tables, allocate KV cache
 * ================================================================ */

extern "C"
void shirley_attn_params_init(
    struct shirley_attn_params * p,
    int n_embd, int n_head, int n_kv_head, int head_dim,
    int max_seq_len, float eps, float kq_scale, int layer_idx,
    float rope_freq_base,
    const struct ggml_tensor * wq,
    const struct ggml_tensor * wk,
    const struct ggml_tensor * wv
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

    /* Weight data */
    p->wq_data = wq->data;
    p->wk_data = wk->data;
    p->wv_data = wv->data;

    /* Weight scales */
    p->wq_wscale = *(const float *)((const uint8_t *)wq->data + (wq->ne[0] * wq->ne[1] / 4));
    p->wk_wscale = *(const float *)((const uint8_t *)wk->data + (wk->ne[0] * wk->ne[1] / 4));
    p->wv_wscale = *(const float *)((const uint8_t *)wv->data + (wv->ne[0] * wv->ne[1] / 4));

    /* Precompute RoPE sin/cos tables — CONST prime values.
     * These are computed ONCE at model load. No runtime transcendentals.
     * inv_freq[i] = 1 / (base ^ (2i / head_dim)) */
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

    /* Workspace */
    int max_dim = n_embd > (n_head * max_seq_len) ? n_embd : (n_head * max_seq_len);
    p->w_raw = (float *)malloc(max_dim * sizeof(float));

    p->ready = 1;

    fprintf(stderr, "shirley: MTFP21 attention init layer %d (n=%d, heads=%d/%d, hd=%d, "
            "rope_base=%.0f, max_seq=%d)\n",
            layer_idx, n_embd, n_head, n_kv_head, head_dim,
            rope_freq_base, max_seq_len);
}
