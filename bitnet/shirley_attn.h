/*
 * shirley_attn.h — Shirley attention: interface declarations
 *
 * The compute function is in shirley_attn.cpp, compiled with AVX2 flags.
 * Covers: QKV matmuls → RoPE → Q@K^T → softmax → attn@V
 * All in MTFP21. Int8 only at ternary matmul wire interface.
 */

#ifndef SHIRLEY_ATTN_H
#define SHIRLEY_ATTN_H

#include <stdint.h>

struct ggml_tensor;

struct shirley_attn_params {
    int n_embd;         /* 2560 */
    int n_head;         /* 20 */
    int n_kv_head;      /* 5 */
    int head_dim;       /* 128 */
    int n_rot;          /* 128 (rotation dimensions) */
    float eps;
    int32_t eps_mant;   /* precomputed MTFP21 */
    int8_t  eps_exp;
    float kq_scale;     /* 1/sqrt(head_dim) — float for init */
    int32_t kq_scale_mant;  /* precomputed MTFP21 */
    int8_t  kq_scale_exp;
    int layer_idx;

    /* QKV + wo weight tensors (2-bit packed ternary) */
    const void * wq_data;
    const void * wk_data;
    const void * wv_data;
    const void * wo_data;

    /* Weight scales */
    float wq_wscale;
    float wk_wscale;
    float wv_wscale;
    float wo_wscale;

    /* Per-layer learned scales */
    float wo_lscale;       /* wo_scale, 1.0 if absent */

    /* Norm gammas — precomputed as MTFP21 at model load */
    int32_t * attn_norm_gamma_mant;
    int8_t  * attn_norm_gamma_exp;
    int32_t * sub_norm_gamma_mant;
    int8_t  * sub_norm_gamma_exp;

    /* RoPE sin/cos tables — precomputed as MTFP21 (CONST prime).
     * Layout: [max_seq_len][head_dim/2] pairs.
     * Mantissa and exponent stored in parallel arrays for cache efficiency. */
    int32_t * rope_cos_mant;  /* [max_seq_len * head_dim/2] */
    int8_t  * rope_cos_exp;   /* [max_seq_len * head_dim/2] */
    int32_t * rope_sin_mant;  /* [max_seq_len * head_dim/2] */
    int8_t  * rope_sin_exp;   /* [max_seq_len * head_dim/2] */
    int max_seq_len;
    float rope_freq_base;

    /* KV cache — native MTFP21 per element.
     * Layout: [max_seq_len][n_kv_head * head_dim] */
    int32_t * k_cache_mant;
    int8_t  * k_cache_exp;
    int32_t * v_cache_mant;
    int8_t  * v_cache_exp;
    int kv_pos;          /* current write position in cache */
    int kv_len;          /* total valid entries in cache */

    /* Pre-allocated workspace */
    float * w_raw;       /* [max(n_embd, n_head * max_seq_len)] */

    /* Split-node shared state: written by one node, read by the next */
    int16_t * mt_act;         /* block-aligned activations (prep → qkv) */
    int8_t    mt_bexp;
    void    * mt_qkv;         /* mtfp21_t[n_embd + kv_dim + kv_dim] (qkv → body) */
    int16_t * mt_wo_act;      /* block-aligned for wo (body → wo) */
    int8_t    mt_wo_bexp;
    void    * mt_wo_out;      /* mtfp21_t[n_embd] (wo → finish) */
    const float * mt_input;   /* current token's ggml input ptr (prep → finish) */
    float *   mt_output;      /* current token's ggml output ptr (finish writes) */
    int       mt_pos;         /* current position for RoPE + cache */

    int ready;
};

#ifdef __cplusplus
extern "C" {
#endif

/* The attention compute function — called by ggml_map_custom1.
 * Input: normalized activations (after attn_norm) [n_embd, n_tokens]
 * Output: attention context [n_embd, n_tokens] (before attn_sub_norm)
 *
 * For single-token generation: uses KV cache.
 * For prompt eval: processes all tokens, fills cache. */
/* Split-node attention: 5 callbacks, each a separate ggml graph node.
 * ggml manages thread barriers between nodes. No spin-waits. */
void shirley_attn_prep(struct ggml_tensor * dst, const struct ggml_tensor * a, int ith, int nth, void * userdata);
void shirley_attn_qkv(struct ggml_tensor * dst, const struct ggml_tensor * a, int ith, int nth, void * userdata);
void shirley_attn_body(struct ggml_tensor * dst, const struct ggml_tensor * a, int ith, int nth, void * userdata);
void shirley_attn_wo(struct ggml_tensor * dst, const struct ggml_tensor * a, int ith, int nth, void * userdata);
void shirley_attn_finish(struct ggml_tensor * dst, const struct ggml_tensor * a, int ith, int nth, void * userdata);

/* Initialize per-layer params from model weights.
 * Also precomputes RoPE sin/cos tables and allocates KV cache. */
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
    const struct ggml_tensor * attn_sub_norm);

#ifdef __cplusplus
}
#endif

#endif /* SHIRLEY_ATTN_H */
