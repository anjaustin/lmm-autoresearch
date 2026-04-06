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
#include "shirley_mtfp16_ops.h"

#include "shirley_convert.h"

#include "3rdparty/llama.cpp/ggml/include/ggml.h"
#include "shirley_ffn.h"
#include "shirley_profile.h"

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
    const int32_t * gamma_mant, const int8_t * gamma_exp,
    int n, int32_t eps_mant, int8_t eps_exp
) {
    /* 1. Sum of squares — chunked SIMD dot product: src · src */
    int32_t s_mant[n]; /* VLA */
    int8_t  s_exp[n];  /* VLA */
    for (int i = 0; i < n; i++) { s_mant[i] = src[i].mantissa; s_exp[i] = src[i].exponent; }
    mtfp21_t sum_sq = mtfp21_dot_chunked(s_mant, s_exp, s_mant, s_exp, n);

    /* 2. rsqrt — one scalar (irreducible) */
    mtfp21_t mean = mtfp21_div_scalar(sum_sq, n);
    mtfp21_t scale = mtfp21_rsqrt(mtfp21_add(mean, (mtfp21_t){eps_mant, eps_exp}));

    /* 3. Per-element scale × gamma — SIMD multiply
     * scale is a scalar broadcast. Multiply all mantissas by scale.mantissa,
     * add scale.exponent to all exponents. Then multiply by gamma. */
    int32_t scale_m = scale.mantissa;
    int8_t  scale_e = scale.exponent;

    mtfp21_t normed[n]; /* VLA */
    int i;
    for (i = 0; i + 8 <= n; i += 8) {
        /* Load 8 source mantissas */
        __m256i vm = _mm256_loadu_si256((const __m256i *)(s_mant + i));
        /* Multiply by scale mantissa: int32 × int32 → int64 (need 4-lane) */
        __m256i vs = _mm256_set1_epi32(scale_m);
        /* Use mul_epi32 for lanes 0,2,4,6 then shift for 1,3,5,7 */
        __m256i prod_even = _mm256_mul_epi32(vm, vs);
        __m256i vm_odd = _mm256_srli_epi64(vm, 32);
        __m256i prod_odd = _mm256_mul_epi32(vm_odd, vs);

        int64_t prods[8];
        int64_t even[4], odd[4];
        _mm256_storeu_si256((__m256i *)even, prod_even);
        _mm256_storeu_si256((__m256i *)odd, prod_odd);
        prods[0] = even[0]; prods[1] = odd[0];
        prods[2] = even[1]; prods[3] = odd[1];
        prods[4] = even[2]; prods[5] = odd[2];
        prods[6] = even[3]; prods[7] = odd[3];

        for (int j = 0; j < 8; j++) {
            int exp = (int)s_exp[i+j] + (int)scale_e;
            /* Normalize product to MTFP21 range */
            int64_t p = prods[j];
            while (llabs(p) > MTFP21_MANT_MAX) {
                p /= 3; exp++;
            }
            while (p != 0 && llabs(p) * 3 <= MTFP21_MANT_MAX && exp > -MTFP21_EXP_MAX) {
                p *= 3; exp--;
            }
            normed[i+j].mantissa = (int32_t)p;
            normed[i+j].exponent = (int8_t)exp;

            if (gamma_mant) {
                mtfp21_t g = {gamma_mant[i+j], gamma_exp[i+j]};
                normed[i+j] = mtfp21_mul(normed[i+j], g);
            }
        }
    }
    /* Scalar tail */
    for (; i < n; i++) {
        normed[i] = mtfp21_mul(src[i], scale);
        if (gamma_mant) {
            mtfp21_t g = {gamma_mant[i], gamma_exp[i]};
            normed[i] = mtfp21_mul(normed[i], g);
        }
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

/* shirley_gemv_mtfp16_part is now in shirley_mtfp16_matmul.h */

extern "C"
void shirley_ffn_compute(
    struct ggml_tensor * dst,
    const struct ggml_tensor * a,
    int ith, int nth,
    void * userdata
) {
    struct shirley_ffn_params * p = (struct shirley_ffn_params *)userdata;
    const int n = p->n_embd;
    const int n_ff = p->n_ff;
    const int n_tokens = (int)a->ne[1];
    float * output = (float *)dst->data;

    for (int tok = 0; tok < n_tokens; tok++) {
        const float * input = (const float *)a->data + tok * n;
        float * out_tok = output + tok * n;

        SP_START; /* starts timer for all threads, only thread 0 uses SP_LAP */

        /* ---- PHASE 0: Sequential prep (thread 0 only) ---- */
        if (ith == 0) {

            mtfp21_t inp_m[n]; /* VLA */
            for (int i = 0; i < n; i++) inp_m[i] = mtfp21_from_float(input[i]);
            SP_LAP(ffn_input_conv);

            /* Store inp_m for residual later (in output buffer temporarily) */
            /* We'll read input[] again for residual at the end */

            p->mt_bexp = mtfp21_rmsnorm_to_mtfp16(
                p->mt_act, inp_m, p->ffn_norm_gamma_mant, p->ffn_norm_gamma_exp, n, p->eps_mant, p->eps_exp);
            SP_LAP(ffn_norm);

            __atomic_store_n(&p->mt_phase, 1, __ATOMIC_RELEASE); /* signal: gate+up ready */
        } else {
            while (__atomic_load_n(&p->mt_phase, __ATOMIC_ACQUIRE) < 1) { /* spin */ }
        }

        /* ---- PHASE 1: Gate + Up matmul — raw int32 output (ALL threads) ---- */
        shirley_gemv_mtfp16_part_raw(p->mt_gate_raw, p->mt_act,
            p->gate_data, n, n_ff, ith, nth);
        shirley_gemv_mtfp16_part_raw(p->mt_up_raw, p->mt_act,
            p->up_data, n, n_ff, ith, nth);

        __atomic_fetch_add(&p->mt_threads_done, 1, __ATOMIC_ACQ_REL);
        if (ith == 0) {
            while (__atomic_load_n(&p->mt_threads_done, __ATOMIC_ACQUIRE) < nth) { /* spin */ }
            __atomic_store_n(&p->mt_threads_done, 0, __ATOMIC_RELEASE);
            SP_LAP(ffn_gate_up);

            /* ---- PHASE 2: Fused trivials — deferred normalization ----
             * raw int32 gate/up → relu(sign check) → square(int64) → multiply(int64)
             * → ONE normalization to MTFP21. No intermediate trit-shift loops.
             *
             * Weight scales applied as MTFP21 multiply after normalization. */
            float gate_ws = p->gate_wscale * p->gate_lscale;
            float up_ws   = p->up_wscale * p->up_lscale;
            /* Combined scale for the fused result:
             * real = gate_raw × gate_ws × gate_ws × up_raw × up_ws / act_scale³
             * (gate_ws appears twice because of square)
             * But we track exponents: block_exp contributes to both gate and up.
             * gate_real = gate_raw × 3^block_exp × gate_ws
             * gate²_real = gate_raw² × 3^(2×block_exp) × gate_ws²
             * result_real = gate²_real × up_real = gate_raw² × up_raw × 3^(3×block_exp) × gate_ws² × up_ws */
            mtfp21_t combined_ws = mtfp21_mul(
                mtfp21_mul(mtfp21_from_float(gate_ws), mtfp21_from_float(gate_ws)),
                mtfp21_from_float(up_ws));
            int combined_exp = (int)p->mt_bexp * 3; /* 3× because gate² × up uses block_exp three times */

            mtfp21_t ffn_out[n_ff]; /* VLA */
            for (int i = 0; i < n_ff; i++) {
                int32_t g = p->mt_gate_raw[i];
                int32_t u = p->mt_up_raw[i];

                /* ReLU: check sign. No normalization. */
                if (g <= 0) { ffn_out[i] = (mtfp21_t){0, 0}; continue; }

                /* Square × Up: int32 × int32 → int64, then × int32 → int64
                 * g² × u. Max: 75M² × 75M = 4.2e23. Fits int64 (max 9.2e18)...
                 * NO — 75M² = 5.6e15, × 75M = 4.2e23. Does NOT fit int64 (max 9.2e18).
                 * Need to scale down before multiplying. */
                int64_t g_sq = (int64_t)g * (int64_t)g;
                /* Normalize g_sq to fit before multiplying by u */
                int extra_exp = 0;
                while (llabs(g_sq) > 2000000000LL) { /* keep within safe int64 range for × u */
                    g_sq /= 3;
                    extra_exp++;
                }
                int64_t result = g_sq * (int64_t)u;

                /* Normalize to MTFP21 — ONE normalization for the entire trivials */
                int exp = combined_exp + extra_exp;
                while (llabs(result) > MTFP21_MANT_MAX) {
                    int64_t rem = result % 3; result /= 3;
                    if (rem == 2) result++; else if (rem == -2) result--;
                    exp++;
                }
                while (result != 0 && llabs(result) * 3 <= MTFP21_MANT_MAX && exp > -MTFP21_EXP_MAX) {
                    result *= 3; exp--;
                }

                mtfp21_t r; r.mantissa = (int32_t)result; r.exponent = (int8_t)exp;
                ffn_out[i] = mtfp21_mul(r, combined_ws);
            }
            SP_LAP(ffn_trivials);

            /* ---- PHASE 3: Sub-norm (thread 0 only) ---- */
            p->mt_sub_bexp = mtfp21_rmsnorm_to_mtfp16(
                p->mt_sub_act, ffn_out, p->ffn_sub_norm_gamma_mant, p->ffn_sub_norm_gamma_exp, n_ff, p->eps_mant, p->eps_exp);
            SP_LAP(ffn_sub_norm);

            __atomic_store_n(&p->mt_phase, 3, __ATOMIC_RELEASE); /* signal: down ready */
        } else {
            while (__atomic_load_n(&p->mt_phase, __ATOMIC_ACQUIRE) < 3) { /* spin */ }
        }

        /* ---- PHASE 4: Down matmul (ALL threads) ---- */
        shirley_gemv_mtfp16_part(((mtfp21_t*)p->mt_down), p->mt_sub_act, p->mt_sub_bexp,
            p->down_data, n_ff, n, p->down_wscale * p->down_lscale, ith, nth);

        __atomic_fetch_add(&p->mt_threads_done, 1, __ATOMIC_ACQ_REL);
        if (ith == 0) {
            while (__atomic_load_n(&p->mt_threads_done, __ATOMIC_ACQUIRE) < nth) { /* spin */ }
            __atomic_store_n(&p->mt_threads_done, 0, __ATOMIC_RELEASE);
            SP_LAP(ffn_down);

            /* ---- PHASE 5: Residual + output (thread 0 only) ---- */
            for (int i = 0; i < n; i++) {
                mtfp21_t inp_i = mtfp21_from_float(input[i]);
                out_tok[i] = mtfp21_to_float(mtfp21_add(((mtfp21_t*)p->mt_down)[i], inp_i));
            }
            SP_LAP(ffn_residual);
            SP_TOKEN();

            __atomic_store_n(&p->mt_phase, 0, __ATOMIC_RELEASE); /* reset for next token */
        } else {
            while (__atomic_load_n(&p->mt_phase, __ATOMIC_ACQUIRE) != 0) { /* spin */ }
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
    { mtfp21_t e = mtfp21_from_float(eps); p->eps_mant = e.mantissa; p->eps_exp = e.exponent; }
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

    shirley_convert_f32_to_mtfp21(
        &p->ffn_norm_gamma_mant, &p->ffn_norm_gamma_exp,
        ffn_norm ? (const float *)ffn_norm->data : NULL, n_embd);
    shirley_convert_f32_to_mtfp21(
        &p->ffn_sub_norm_gamma_mant, &p->ffn_sub_norm_gamma_exp,
        ffn_sub_norm ? (const float *)ffn_sub_norm->data : NULL, n_ff);

    int max_dim = n_embd > n_ff ? n_embd : n_ff;
    p->w_act     = (int8_t  *)malloc(max_dim * sizeof(int8_t));
    p->w_gate    = (int8_t  *)malloc(n_ff * sizeof(int8_t));
    p->w_up      = (int8_t  *)malloc(n_ff * sizeof(int8_t));
    p->w_ffn_out = (int8_t  *)malloc(n_ff * sizeof(int8_t));
    p->w_sub     = (int8_t  *)malloc(n_ff * sizeof(int8_t));
    p->w_raw     = (float   *)malloc(max_dim * sizeof(float));
    p->w_sq      = (int16_t *)malloc(n_ff * sizeof(int16_t));
    p->w_prod    = (int32_t *)malloc(n_ff * sizeof(int32_t));

    /* Threading workspace */
    p->mt_phase = 0;
    p->mt_threads_done = 0;
    p->mt_act     = (int16_t  *)malloc(max_dim * sizeof(int16_t));
    p->mt_gate    = malloc(n_ff * sizeof(mtfp21_t));
    p->mt_up      = malloc(n_ff * sizeof(mtfp21_t));
    p->mt_down    = malloc(n_embd * sizeof(mtfp21_t));
    p->mt_gate_raw = (int32_t *)malloc(n_ff * sizeof(int32_t));
    p->mt_up_raw   = (int32_t *)malloc(n_ff * sizeof(int32_t));
    p->mt_sub_act = (int16_t  *)malloc(n_ff * sizeof(int16_t));

    p->ready = 1;

    fprintf(stderr, "shirley: MTFP16 FFN init layer %d (n=%d, n_ff=%d)\n",
            layer_idx, n_embd, n_ff);
}
