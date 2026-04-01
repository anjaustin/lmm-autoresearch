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

/* Forward declarations */
static inline mtfp21_t mtfp21_mul(mtfp21_t a, mtfp21_t b);
static inline mtfp21_t mtfp21_add(mtfp21_t a, mtfp21_t b);

/* ================================================================
 *  int8 ↔ MTFP21 conversion
 * ================================================================ */

/* Convert int8 activation to MTFP21. The int8 value IS the mantissa. */
static inline mtfp21_t mtfp21_from_int8(int8_t val) {
    mtfp21_t r;
    r.mantissa = (int32_t)val;
    r.exponent = 0;
    return r;
}

/* Convert MTFP21 to int8, clamped to [-max_range, +max_range].
 * max_range: 40 for 4-trit, 121 for 5-trit, 1093 for 7-trit, etc. */
static inline int8_t mtfp21_to_int8(mtfp21_t a, int max_range) {
    if (a.mantissa == 0) return 0;

    int64_t val;
    int exp = a.exponent;

    if (exp == 0) {
        val = (int64_t)a.mantissa;
    } else if (exp > 0) {
        /* Multiply mantissa by 3^exp, guarding against int64 overflow */
        int64_t abs_m = (int64_t)(a.mantissa > 0 ? a.mantissa : -a.mantissa);
        if (exp >= 32 || abs_m > INT64_MAX / POW3[exp]) {
            val = (a.mantissa > 0) ? max_range : -max_range;
            goto clamp;
        }
        val = (int64_t)a.mantissa * POW3[exp];
    } else {
        /* Divide mantissa by 3^(-exp), with rounding */
        int neg_exp = -exp;
        if (neg_exp < 32) {
            int64_t divisor = POW3[neg_exp];
            int64_t half = divisor / 2;
            if (a.mantissa >= 0) {
                val = ((int64_t)a.mantissa + half) / divisor;
            } else {
                val = -(((int64_t)(-a.mantissa) + half) / divisor);
            }
        } else {
            val = 0; /* Underflow to zero */
        }
    }

clamp:
    if (val > max_range)  val = max_range;
    if (val < -max_range) val = -max_range;
    return (int8_t)val;
}

/* ================================================================
 *  Integer division
 * ================================================================ */

/* Compute 1/n as MTFP21 without using float.
 * Finds exponent E such that POW3[-E] / n maximizes mantissa precision. */
static inline mtfp21_t mtfp21_recip_int(int32_t n) {
    mtfp21_t r;
    if (n == 0) { r.mantissa = 0; r.exponent = MTFP21_EXP_MAX; return r; }
    if (n == 1) { r.mantissa = 1; r.exponent = 0; return r; }
    if (n == -1) { r.mantissa = -1; r.exponent = 0; return r; }

    int sign = (n > 0) ? 1 : -1;
    int64_t absn = (int64_t)(n > 0 ? n : -n);

    /* Search for the largest exponent magnitude where POW3[e] / absn fits in mantissa */
    int best_e = 0;
    int64_t best_mant = 0;

    for (int e = 0; e < 31; e++) {
        int64_t mant = POW3[e] / absn;
        if (mant > MTFP21_MANT_MAX) break;
        /* Round: check if remainder >= half divisor */
        int64_t rem = POW3[e] % absn;
        if (rem * 2 >= absn) mant++;
        if (mant <= MTFP21_MANT_MAX && mant > best_mant) {
            best_mant = mant;
            best_e = e;
        }
    }

    r.mantissa = (int32_t)(sign * best_mant);
    r.exponent = (int8_t)(-best_e);
    return r;
}

/* Divide MTFP21 value by integer scalar. No float operations. */
static inline mtfp21_t mtfp21_div_scalar(mtfp21_t a, int32_t n) {
    if (a.mantissa == 0) return a;
    mtfp21_t recip = mtfp21_recip_int(n);
    return mtfp21_mul(a, recip);
}

/* General MTFP21 / MTFP21 division via reciprocal multiply. */
static inline mtfp21_t mtfp21_div(mtfp21_t a, mtfp21_t b) {
    if (a.mantissa == 0) { mtfp21_t z = {0, 0}; return z; }
    if (b.mantissa == 0) { mtfp21_t z = {0, 0}; return z; } /* div by zero → 0 */

    /* Compute reciprocal of b: find M,E such that M * 3^E ≈ 1 / (b.mantissa * 3^b.exponent)
     * = 1/b.mantissa * 3^(-b.exponent)
     * We want M = MTFP21_MANT_MAX-scale / |b.mantissa| with appropriate exponent. */
    int sign_b = (b.mantissa > 0) ? 1 : -1;
    int64_t abs_bm = (int64_t)(b.mantissa > 0 ? b.mantissa : -b.mantissa);

    /* Find e such that POW3[e] / abs_bm is close to MTFP21_MANT_MAX */
    int best_e = 0;
    int64_t best_mant = 0;
    for (int e = 0; e < 31; e++) {
        int64_t mant = POW3[e] / abs_bm;
        if (mant > MTFP21_MANT_MAX) break;
        int64_t rem = POW3[e] % abs_bm;
        if (rem * 2 >= abs_bm) mant++;
        if (mant <= MTFP21_MANT_MAX && mant > best_mant) {
            best_mant = mant;
            best_e = e;
        }
    }

    mtfp21_t recip;
    recip.mantissa = (int32_t)(sign_b * best_mant);
    recip.exponent = (int8_t)(-best_e - b.exponent);

    return mtfp21_mul(a, recip);
}

/* ================================================================ */

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

/* 256-entry rsqrt LUT for integer-only Newton-Raphson seed.
 * Indexed by normalized mantissa in [MANT_MAX/3, MANT_MAX].
 * Max LUT error: 6.39e-08 (already within float32 precision).
 * Generated by build-time script — runtime never touches float. */
static const struct { int32_t mant; int8_t exp; } RSQRT_LUT[256] = {
    {11693018,-22},{11647784,-22},{11603071,-22},{11558868,-22},{11515167,-22},
    {11471958,-22},{11429232,-22},{11386980,-22},{11345192,-22},{11303862,-22},
    {11262980,-22},{11222538,-22},{11182529,-22},{11142945,-22},{11103778,-22},
    {11065021,-22},{11026667,-22},{10988710,-22},{10951141,-22},{10913956,-22},
    {10877146,-22},{10840707,-22},{10804631,-22},{10768913,-22},{10733547,-22},
    {10698527,-22},{10663848,-22},{10629504,-22},{10595489,-22},{10561799,-22},
    {10528429,-22},{10495372,-22},{10462625,-22},{10430183,-22},{10398041,-22},
    {10366193,-22},{10334637,-22},{10303367,-22},{10272380,-22},{10241670,-22},
    {10211234,-22},{10181068,-22},{10151168,-22},{10121529,-22},{10092149,-22},
    {10063023,-22},{10034147,-22},{10005519,-22},{ 9977135,-22},{ 9948990,-22},
    { 9921083,-22},{ 9893409,-22},{ 9865965,-22},{ 9838749,-22},{ 9811756,-22},
    { 9784985,-22},{ 9758431,-22},{ 9732093,-22},{ 9705966,-22},{ 9680049,-22},
    { 9654338,-22},{ 9628831,-22},{ 9603526,-22},{ 9578418,-22},{ 9553507,-22},
    { 9528789,-22},{ 9504262,-22},{ 9479923,-22},{ 9455770,-22},{ 9431801,-22},
    { 9408013,-22},{ 9384405,-22},{ 9360973,-22},{ 9337716,-22},{ 9314631,-22},
    { 9291717,-22},{ 9268971,-22},{ 9246391,-22},{ 9223976,-22},{ 9201723,-22},
    { 9179630,-22},{ 9157695,-22},{ 9135917,-22},{ 9114293,-22},{ 9092823,-22},
    { 9071503,-22},{ 9050333,-22},{ 9029310,-22},{ 9008433,-22},{ 8987700,-22},
    { 8967109,-22},{ 8946660,-22},{ 8926350,-22},{ 8906177,-22},{ 8886141,-22},
    { 8866239,-22},{ 8846470,-22},{ 8826833,-22},{ 8807326,-22},{ 8787948,-22},
    { 8768697,-22},{ 8749573,-22},{ 8730572,-22},{ 8711695,-22},{ 8692940,-22},
    { 8674306,-22},{ 8655791,-22},{ 8637394,-22},{ 8619114,-22},{ 8600949,-22},
    { 8582899,-22},{ 8564962,-22},{ 8547137,-22},{ 8529422,-22},{ 8511818,-22},
    { 8494322,-22},{ 8476933,-22},{ 8459651,-22},{ 8442474,-22},{ 8425401,-22},
    { 8408431,-22},{ 8391564,-22},{ 8374798,-22},{ 8358131,-22},{ 8341564,-22},
    { 8325095,-22},{ 8308723,-22},{ 8292448,-22},{ 8276267,-22},{ 8260181,-22},
    { 8244188,-22},{ 8228288,-22},{ 8212480,-22},{ 8196762,-22},{ 8181134,-22},
    { 8165596,-22},{ 8150145,-22},{ 8134782,-22},{ 8119505,-22},{ 8104315,-22},
    { 8089209,-22},{ 8074187,-22},{ 8059249,-22},{ 8044393,-22},{ 8029619,-22},
    { 8014926,-22},{ 8000314,-22},{ 7985781,-22},{ 7971327,-22},{ 7956951,-22},
    { 7942653,-22},{ 7928431,-22},{ 7914286,-22},{ 7900216,-22},{ 7886221,-22},
    { 7872300,-22},{ 7858453,-22},{ 7844678,-22},{ 7830976,-22},{ 7817345,-22},
    { 7803785,-22},{ 7790295,-22},{ 7776875,-22},{ 7763524,-22},{ 7750242,-22},
    { 7737027,-22},{ 7723880,-22},{ 7710800,-22},{ 7697786,-22},{ 7684838,-22},
    { 7671954,-22},{ 7659136,-22},{ 7646381,-22},{ 7633690,-22},{ 7621062,-22},
    { 7608496,-22},{ 7595993,-22},{ 7583550,-22},{ 7571169,-22},{ 7558848,-22},
    { 7546587,-22},{ 7534386,-22},{ 7522243,-22},{ 7510159,-22},{ 7498134,-22},
    { 7486165,-22},{ 7474254,-22},{ 7462400,-22},{ 7450601,-22},{ 7438859,-22},
    { 7427172,-22},{ 7415539,-22},{ 7403962,-22},{ 7392438,-22},{ 7380968,-22},
    { 7369551,-22},{ 7358187,-22},{ 7346876,-22},{ 7335616,-22},{ 7324408,-22},
    { 7313251,-22},{ 7302145,-22},{ 7291090,-22},{ 7280084,-22},{ 7269129,-22},
    { 7258222,-22},{ 7247365,-22},{ 7236556,-22},{ 7225795,-22},{ 7215082,-22},
    { 7204417,-22},{ 7193799,-22},{ 7183228,-22},{21518108,-23},{21486672,-23},
    {21455373,-23},{21424210,-23},{21393183,-23},{21362291,-23},{21331531,-23},
    {21300904,-23},{21270409,-23},{21240045,-23},{21209810,-23},{21179703,-23},
    {21149725,-23},{21119874,-23},{21090148,-23},{21060548,-23},{21031072,-23},
    {21001720,-23},{20972490,-23},{20943381,-23},{20914394,-23},{20885527,-23},
    {20856778,-23},{20828149,-23},{20799637,-23},{20771241,-23},{20742962,-23},
    {20714797,-23},{20686748,-23},{20658811,-23},{20630988,-23},{20603277,-23},
    {20575677,-23},{20548188,-23},{20520808,-23},{20493538,-23},{20466376,-23},
    {20439322,-23},{20412375,-23},{20385535,-23},{20358799,-23},{20332169,-23},
    {20305643,-23}
};

/* Integer-only rsqrt: LUT seed + 2 Newton-Raphson iterations.
 * No float operations.
 *
 * For input x = M * 3^E:
 *   1/sqrt(x) = 1/sqrt(M) * 3^(-E/2)
 *
 * If E is even: 3^(-E/2) is a clean exponent shift.
 * If E is odd:  E = 2k+1, so 3^(-E/2) = 3^(-k) * 3^(-1/2) = 3^(-k) / sqrt(3).
 *               We multiply by INV_SQRT3 = {8284027, -15} ≈ 1/sqrt(3).
 */
static inline mtfp21_t mtfp21_rsqrt(mtfp21_t x) {
    if (x.mantissa <= 0) {
        mtfp21_t r = {0, 0};
        return r;
    }

    /* Normalize mantissa to [MANT_MAX/3, MANT_MAX] */
    int32_t mant = x.mantissa;
    int exp = x.exponent;
    while (mant > 0 && (int64_t)mant * 3 <= MTFP21_MANT_MAX && exp > -MTFP21_EXP_MAX) {
        mant *= 3;
        exp--;
    }

    /* LUT index from normalized mantissa */
    int32_t mant_min = MTFP21_MANT_MAX / 3; /* 7174453 */
    int32_t mant_range = MTFP21_MANT_MAX - mant_min; /* 14348907 */
    int idx;
    if (mant <= mant_min) {
        idx = 0;
    } else {
        idx = (int)((int64_t)(mant - mant_min) * 256 / mant_range);
        if (idx > 255) idx = 255;
    }

    /* Seed from LUT: gives 1/sqrt(mant) for the normalized mantissa */
    mtfp21_t y;
    y.mantissa = RSQRT_LUT[idx].mant;
    y.exponent = RSQRT_LUT[idx].exp;

    /* Apply exponent factor: need 1/sqrt(mant * 3^exp) = 1/sqrt(mant) * 3^(-exp/2) */
    if (exp % 2 == 0) {
        /* Even exponent: clean shift */
        y.exponent += (-exp / 2);
    } else {
        /* Odd exponent: E = 2k + 1, k = (E-1)/2
         * 3^(-E/2) = 3^(-k) * 3^(-1/2) = 3^(-k) / sqrt(3) */
        static const mtfp21_t INV_SQRT3 = {8284027, -15};
        int k = (exp - 1) / 2;
        y.exponent += (-k);
        y = mtfp21_mul(y, INV_SQRT3);
    }

    /* Newton-Raphson: y_new = y * (3 - x*y*y) / 2
     * Constants computed without float:
     *   three = 14348907 * 3^(-14) = 3^15 / 3^14 = 3 (exact, fully normalized)
     *   half  = 21523360 * 3^(-16) ≈ 0.5 (error 1.16e-8)
     * CRITICAL: three must be fully normalized ({14348907,-14} not {1,1})
     * so that exponent alignment in the subtraction 3 - x*y^2 doesn't
     * destroy precision. {1,1} has exp=1 while working values have exp~-15,
     * causing 16 positions of alignment shift. */
    static const mtfp21_t NR_THREE = {14348907, -14};
    static const mtfp21_t NR_HALF  = {21523360, -16};

    for (int i = 0; i < 2; i++) {
        mtfp21_t y2   = mtfp21_mul(y, y);
        mtfp21_t xy2  = mtfp21_mul(x, y2);
        mtfp21_t diff = mtfp21_add(NR_THREE, mtfp21_neg(xy2));
        mtfp21_t step = mtfp21_mul(diff, NR_HALF);
        y = mtfp21_mul(y, step);
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

    /* Step 2: mean = sum / n (integer division — no float) */
    mtfp21_t mean = mtfp21_div_scalar(sum_sq, n);

    /* Step 3: add epsilon */
    mtfp21_t eps_m = mtfp21_from_float(eps);
    mtfp21_t mean_eps = mtfp21_add(mean, eps_m);

    /* Step 4: rsqrt (integer-only LUT + NR) */
    mtfp21_t scale = mtfp21_rsqrt(mean_eps);

    /* Step 5: apply scale via MTFP21 multiply, convert to float at output */
    for (int i = 0; i < n; i++) {
        mtfp21_t xi = mtfp21_from_float(src[i]);
        mtfp21_t yi = mtfp21_mul(xi, scale);
        dst[i] = mtfp21_to_float(yi);
    }
}

/* ================================================================
 *  Pure MTFP21 RMSNorm — zero float between input and output conversion
 * ================================================================ */

static inline void mtfp21_rmsnorm_pure(
    mtfp21_t * restrict dst,
    const mtfp21_t * restrict src,
    int n,
    mtfp21_t eps
) {
    /* Sum of squares */
    mtfp21_t sum_sq = {0, 0};
    for (int i = 0; i < n; i++) {
        mtfp21_t xi2 = mtfp21_mul(src[i], src[i]);
        sum_sq = mtfp21_add(sum_sq, xi2);
    }

    /* mean = sum / n */
    mtfp21_t mean = mtfp21_div_scalar(sum_sq, n);

    /* rsqrt(mean + eps) */
    mtfp21_t mean_eps = mtfp21_add(mean, eps);
    mtfp21_t scale = mtfp21_rsqrt(mean_eps);

    /* Apply scale */
    for (int i = 0; i < n; i++) {
        dst[i] = mtfp21_mul(src[i], scale);
    }
}

/* ================================================================
 *  int8-to-int8 RMSNorm via MTFP21 — the actual pipeline operation
 *
 *  Input:  int8 activations (e.g. 5-trit range [-80, +80])
 *  Output: int8 normalized activations (7-trit range for RMSNorm)
 * ================================================================ */

static inline void mtfp21_rmsnorm_int8(
    int8_t * restrict dst,
    const int8_t * restrict src,
    int n,
    mtfp21_t eps,
    int out_range,          /* output clamp range, e.g. 121 for 5-trit */
    mtfp21_t *work_src,     /* caller-provided workspace, n elements */
    mtfp21_t *work_dst      /* caller-provided workspace, n elements */
) {
    for (int i = 0; i < n; i++) {
        work_src[i] = mtfp21_from_int8(src[i]);
    }

    mtfp21_rmsnorm_pure(work_dst, work_src, n, eps);

    for (int i = 0; i < n; i++) {
        dst[i] = mtfp21_to_int8(work_dst[i], out_range);
    }
}

#endif /* SHIRLEY_MTFP21_H */
