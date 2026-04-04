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
            act_mant, inp_m, p->ffn_norm_gamma_mant, p->ffn_norm_gamma_exp, n, p->eps_mant, p->eps_exp);

        /* 2. Gate matmul: MTFP16 × ternary → MTFP21 (pure integer, sign_epi16) */
        mtfp21_t gate_m[n_ff]; /* VLA */
        shirley_gemv_mtfp16(gate_m, act_mant, block_exp,
            p->gate_data, n, n_ff, p->gate_wscale * p->gate_lscale);

        /* 3. Up matmul: same path */
        mtfp21_t up_m[n_ff]; /* VLA */
        shirley_gemv_mtfp16(up_m, act_mant, block_exp,
            p->up_data, n, n_ff, p->up_wscale * p->up_lscale);

        /* 4-6. ReLU → Square → gate² × up (SIMD where possible) */

        /* Convert gate and up from MTFP21 to MTFP16 parallel arrays */
        int16_t gate_mant16[n_ff], up_mant16[n_ff]; /* VLA */
        int8_t gate_exp16[n_ff], up_exp16[n_ff]; /* VLA */
        for (int i = 0; i < n_ff; i++) {
            mtfp16_local_t g16 = to_mtfp16(gate_m[i]);
            gate_mant16[i] = g16.mantissa;
            gate_exp16[i] = g16.exponent;
            mtfp16_local_t u16 = to_mtfp16(up_m[i]);
            up_mant16[i] = u16.mantissa;
            up_exp16[i] = u16.exponent;
        }

        /* 4. ReLU — SIMD 16 lanes */
        mtfp16_relu_simd(gate_mant16, gate_exp16, gate_mant16, gate_exp16, n_ff);

        /* 5. Square — SIMD 8 lanes (widen to int32) */
        int32_t sq_mant32[n_ff]; /* VLA */
        int8_t sq_exp8[n_ff]; /* VLA */
        mtfp16_square_simd(sq_mant32, sq_exp8, gate_mant16, gate_exp16, n_ff);

        /* 6. gate² × up — int32 × int16 → MTFP21 */
        mtfp21_t ffn_out[n_ff]; /* VLA */
        mtfp_elem_mul_32x16(ffn_out, sq_mant32, sq_exp8, up_mant16, up_exp16, n_ff);

        /* 7. Sub-norm → block-aligned MTFP16 for down matmul */
        int16_t sub_mant[n_ff]; /* VLA */
        int8_t sub_exp = mtfp21_rmsnorm_to_mtfp16(
            sub_mant, ffn_out, p->ffn_sub_norm_gamma_mant, p->ffn_sub_norm_gamma_exp, n_ff, p->eps_mant, p->eps_exp);

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

    p->ready = 1;

    fprintf(stderr, "shirley: MTFP16 FFN init layer %d (n=%d, n_ff=%d)\n",
            layer_idx, n_embd, n_ff);
}
