/*
 * shirley_mtfp21.h — Multi-Trit Floating Point (21-trit) arithmetic
 *
 * MTFP21: 16-trit mantissa (int32) + 5-trit exponent (int8)
 * value = mantissa × 3^exponent
 *
 * Mantissa stored as int32 for arithmetic speed.
 * Trit-level representation used only for sign_epi8 matmul operations.
 *
 * IMPORTANT: The MTFP21 doc's multiply spec (sign_epi8 on mantissa trits)
 * is only correct for multiplying by a single ternary weight {-1,0,+1}.
 * General MTFP21 × MTFP21 uses int32 mantissa multiplication.
 */

#ifndef SHIRLEY_MTFP21_H
#define SHIRLEY_MTFP21_H

#include <stdint.h>
#include <math.h>

/* ================================================================
 *  Type definition
 * ================================================================ */

typedef struct {
    int32_t mantissa;  /* range: [-21523360, +21523360] = -(3^16-1)/2 to +(3^16-1)/2 */
    int8_t  exponent;  /* range: [-121, +121] */
} mtfp21_t;

/* Constants */
#define MTFP21_MANT_MAX   21523360   /* (3^16 - 1) / 2 */
#define MTFP21_MANT_BITS  16         /* number of trits in mantissa */
#define MTFP21_EXP_MAX    121        /* (3^5 - 1) / 2 */

/* ================================================================
 *  Precomputed powers of 3
 * ================================================================ */

static const int64_t POW3[32] = {
    1LL, 3LL, 9LL, 27LL, 81LL, 243LL, 729LL, 2187LL,
    6561LL, 19683LL, 59049LL, 177147LL, 531441LL, 1594323LL, 4782969LL, 14348907LL,
    43046721LL, 129140163LL, 387420489LL, 1162261467LL, 3486784401LL, 10460353203LL,
    31381059609LL, 94143178827LL, 282429536481LL, 847288609443LL, 2541865828329LL,
    7625597484987LL, 22876792454961LL, 68630377364883LL, 205891132094649LL, 617673396283947LL
};

/* ================================================================
 *  Conversion: float32 → MTFP21
 * ================================================================ */

static inline mtfp21_t mtfp21_from_float(float f) {
    mtfp21_t r;

    if (f == 0.0f || fabsf(f) < 1e-45f) {
        r.mantissa = 0;
        r.exponent = 0;
        return r;
    }

    /* Find exponent: largest power of 3 such that |f| / 3^exp fits in mantissa */
    double absf = fabs((double)f);

    /* Use log base 3 to estimate exponent */
    int exp = (int)floor(log(absf) / log(3.0) - (MTFP21_MANT_BITS - 1));

    /* Clamp exponent */
    if (exp < -MTFP21_EXP_MAX) exp = -MTFP21_EXP_MAX;
    if (exp > MTFP21_EXP_MAX)  exp = MTFP21_EXP_MAX;

    /* Compute mantissa = round(f / 3^exp) */
    double scale;
    if (exp >= 0 && exp < 32) {
        scale = (double)POW3[exp];
    } else if (exp < 0 && -exp < 32) {
        scale = 1.0 / (double)POW3[-exp];
    } else {
        scale = pow(3.0, (double)exp);
    }

    int64_t mant = (int64_t)round((double)f / scale);

    /* Normalize: shift mantissa to maximize precision */
    while (mant != 0 && llabs(mant) * 3 <= MTFP21_MANT_MAX && exp > -MTFP21_EXP_MAX) {
        mant *= 3;
        exp--;
    }

    /* Clamp mantissa if overflow */
    while (llabs(mant) > MTFP21_MANT_MAX && exp < MTFP21_EXP_MAX) {
        /* Ternary right-shift: divide by 3, round to nearest */
        int64_t rem = mant % 3;
        mant = mant / 3;
        if (rem == 2) mant++;       /* round up */
        else if (rem == -2) mant--; /* round down (negative) */
        exp++;
    }

    /* Final clamp */
    if (mant > MTFP21_MANT_MAX)  mant = MTFP21_MANT_MAX;
    if (mant < -MTFP21_MANT_MAX) mant = -MTFP21_MANT_MAX;
    if (exp > MTFP21_EXP_MAX)    exp = MTFP21_EXP_MAX;
    if (exp < -MTFP21_EXP_MAX)   exp = -MTFP21_EXP_MAX;

    r.mantissa = (int32_t)mant;
    r.exponent = (int8_t)exp;
    return r;
}

/* ================================================================
 *  Conversion: MTFP21 → float32
 * ================================================================ */

static inline float mtfp21_to_float(mtfp21_t a) {
    if (a.mantissa == 0) return 0.0f;

    double scale;
    int exp = a.exponent;
    if (exp >= 0 && exp < 32) {
        scale = (double)POW3[exp];
    } else if (exp < 0 && -exp < 32) {
        scale = 1.0 / (double)POW3[-exp];
    } else {
        scale = pow(3.0, (double)exp);
    }

    return (float)((double)a.mantissa * scale);
}

/* ================================================================
 *  Basic arithmetic
 * ================================================================ */

/* Negate */
static inline mtfp21_t mtfp21_neg(mtfp21_t a) {
    a.mantissa = -a.mantissa;
    return a;
}

/* Absolute value */
static inline mtfp21_t mtfp21_abs(mtfp21_t a) {
    if (a.mantissa < 0) a.mantissa = -a.mantissa;
    return a;
}

/* Ternary right-shift mantissa by n positions (divide by 3^n, round) */
static inline int32_t trit_rshift(int32_t mant, int n) {
    for (int i = 0; i < n; i++) {
        int32_t rem = mant % 3;
        mant = mant / 3;
        if (rem == 2) mant++;
        else if (rem == -2) mant--;
    }
    return mant;
}

/* Add two MTFP21 values */
static inline mtfp21_t mtfp21_add(mtfp21_t a, mtfp21_t b) {
    mtfp21_t r;

    if (a.mantissa == 0) return b;
    if (b.mantissa == 0) return a;

    /* Align exponents: shift smaller mantissa right */
    int diff = (int)a.exponent - (int)b.exponent;

    int64_t ma, mb;
    int exp;

    if (diff >= 0) {
        exp = a.exponent;
        ma = (int64_t)a.mantissa;
        if (diff > MTFP21_MANT_BITS) {
            return a; /* b is negligible */
        }
        mb = (int64_t)trit_rshift(b.mantissa, diff);
    } else {
        exp = b.exponent;
        mb = (int64_t)b.mantissa;
        if (-diff > MTFP21_MANT_BITS) {
            return b; /* a is negligible */
        }
        ma = (int64_t)trit_rshift(a.mantissa, -diff);
    }

    int64_t sum = ma + mb;

    /* Normalize: handle overflow — single-shot division to avoid rounding accumulation */
    if (llabs(sum) > MTFP21_MANT_MAX) {
        int n = 0;
        int64_t test = llabs(sum);
        while (test > MTFP21_MANT_MAX && n < 31) {
            test /= 3;
            n++;
        }
        if (n > 0 && n < 32) {
            int64_t divisor = POW3[n];
            int64_t half = divisor / 2;
            if (sum >= 0) {
                sum = (sum + half) / divisor;
            } else {
                sum = -((-sum + half) / divisor);
            }
            exp += n;
        }
    }

    /* Normalize: remove leading zeros */
    while (sum != 0 && llabs(sum) * 3 <= MTFP21_MANT_MAX && exp > -MTFP21_EXP_MAX) {
        sum *= 3;
        exp--;
    }

    r.mantissa = (int32_t)sum;
    r.exponent = (int8_t)exp;
    return r;
}

/* Multiply two MTFP21 values */
static inline mtfp21_t mtfp21_mul(mtfp21_t a, mtfp21_t b) {
    mtfp21_t r;

    if (a.mantissa == 0 || b.mantissa == 0) {
        r.mantissa = 0;
        r.exponent = 0;
        return r;
    }

    /* Mantissa multiply in int64 */
    int64_t prod = (int64_t)a.mantissa * (int64_t)b.mantissa;
    int exp = (int)a.exponent + (int)b.exponent;

    /* Normalize: compute how many divisions by 3 are needed in one shot
     * to avoid accumulated rounding errors from iterative division */
    if (llabs(prod) > MTFP21_MANT_MAX) {
        /* Find n such that |prod| / 3^n is in [POW3[15], MTFP21_MANT_MAX] */
        int n = 0;
        int64_t test = llabs(prod);
        while (test > MTFP21_MANT_MAX && n < 31) {
            test /= 3;
            n++;
        }

        /* Do the division in one step using the precomputed power */
        if (n > 0 && n < 32) {
            /* Round to nearest: prod / POW3[n] with rounding */
            int64_t divisor = POW3[n];
            int64_t half = divisor / 2;
            if (prod >= 0) {
                prod = (prod + half) / divisor;
            } else {
                prod = -((-prod + half) / divisor);
            }
            exp += n;
        }
    }

    /* Normalize: remove leading zeros (increase precision) */
    while (prod != 0 && llabs(prod) * 3 <= MTFP21_MANT_MAX && exp > -MTFP21_EXP_MAX) {
        prod *= 3;
        exp--;
    }

    /* Clamp */
    if (prod > MTFP21_MANT_MAX)  prod = MTFP21_MANT_MAX;
    if (prod < -MTFP21_MANT_MAX) prod = -MTFP21_MANT_MAX;
    if (exp > MTFP21_EXP_MAX)    exp = MTFP21_EXP_MAX;
    if (exp < -MTFP21_EXP_MAX)   exp = -MTFP21_EXP_MAX;

    r.mantissa = (int32_t)prod;
    r.exponent = (int8_t)exp;
    return r;
}

/* ================================================================
 *  RMSNorm helper: rsqrt via Newton-Raphson
 *
 *  Computes 1/sqrt(x) in MTFP21 arithmetic.
 *  Uses 3 iterations of Newton-Raphson: y = y * (3 - x*y*y) / 2
 *  (adapted for integer: y = y * (3 - x*y*y) >> 1)
 * ================================================================ */

static inline mtfp21_t mtfp21_rsqrt(mtfp21_t x) {
    if (x.mantissa <= 0) {
        mtfp21_t r = {0, 0};
        return r;
    }

    /* Initial estimate via float (will replace with LUT later) */
    float fx = mtfp21_to_float(x);
    float est = 1.0f / sqrtf(fx);
    mtfp21_t y = mtfp21_from_float(est);

    /* Newton-Raphson: y_new = y * (3 - x * y * y) / 2 */
    mtfp21_t three = mtfp21_from_float(3.0f);
    mtfp21_t half  = mtfp21_from_float(0.5f);

    for (int i = 0; i < 3; i++) {
        mtfp21_t y2  = mtfp21_mul(y, y);      /* y^2 */
        mtfp21_t xy2 = mtfp21_mul(x, y2);     /* x * y^2 */
        mtfp21_t diff = mtfp21_add(three, mtfp21_neg(xy2)); /* 3 - x*y^2 */
        mtfp21_t step = mtfp21_mul(diff, half); /* (3 - x*y^2) / 2 */
        y = mtfp21_mul(y, step);               /* y * (3 - x*y^2) / 2 */
    }

    return y;
}

/* ================================================================
 *  RMSNorm in MTFP21
 *
 *  For a vector x of length n:
 *    scale = rsqrt(mean(x^2) + eps)
 *    y[i] = x[i] * scale
 * ================================================================ */

static inline void mtfp21_rmsnorm(
    float * restrict dst,       /* output (float32) */
    const float * restrict src, /* input (float32) */
    int n,
    float eps
) {
    /* Step 1: convert to MTFP21, compute sum of squares */
    mtfp21_t sum_sq = {0, 0};

    for (int i = 0; i < n; i++) {
        mtfp21_t xi = mtfp21_from_float(src[i]);
        mtfp21_t xi2 = mtfp21_mul(xi, xi);
        sum_sq = mtfp21_add(sum_sq, xi2);
    }

    /* Step 2: mean = sum / n */
    mtfp21_t mean = mtfp21_from_float(mtfp21_to_float(sum_sq) / (float)n);
    /* TODO: implement MTFP21 integer division for full integer path */

    /* Step 3: add epsilon */
    mtfp21_t eps_m = mtfp21_from_float(eps);
    mtfp21_t mean_eps = mtfp21_add(mean, eps_m);

    /* Step 4: rsqrt */
    mtfp21_t scale = mtfp21_rsqrt(mean_eps);

    /* Step 5: apply scale and convert back to float */
    float scale_f = mtfp21_to_float(scale);
    for (int i = 0; i < n; i++) {
        dst[i] = src[i] * scale_f;
    }
}

#endif /* SHIRLEY_MTFP21_H */
