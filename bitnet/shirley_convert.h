/*
 * shirley_convert.h — MTFP21 bulk conversion utilities
 *
 * One-time conversions at model load. Zero float at inference.
 */

#ifndef SHIRLEY_CONVERT_H
#define SHIRLEY_CONVERT_H

#include <stdint.h>
#include <stdlib.h>

/* Convert a float32 array to parallel MTFP21 mantissa + exponent arrays.
 * Allocates both arrays. Caller owns the memory. */
static inline void shirley_convert_f32_to_mtfp21(
    int32_t ** out_mant, int8_t ** out_exp,
    const float * src, int n
) {
    if (!src || n <= 0) {
        *out_mant = NULL;
        *out_exp = NULL;
        return;
    }
    *out_mant = (int32_t *)malloc(n * sizeof(int32_t));
    *out_exp  = (int8_t  *)malloc(n * sizeof(int8_t));
    for (int i = 0; i < n; i++) {
        mtfp21_t m = mtfp21_from_float(src[i]);
        (*out_mant)[i] = m.mantissa;
        (*out_exp)[i]  = m.exponent;
    }
}

#endif
