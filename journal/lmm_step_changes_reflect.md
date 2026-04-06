# REFLECT: Step-Changes for Shirley

## Why three times on Node 7 (fusion and deferred normalization):

**Why does fusion feel like betraying native MTFP?** Because it operates on raw int32/int64 instead of mtfp21_t structs. The relu, square, and multiply happen on the accumulator values directly.

**But is the raw int32 NOT MTFP?** It is. An int32 with a known exponent IS an MTFP value. The struct is a convenience. The math is mantissa × 3^exponent regardless of whether it's in a struct or in a register with the exponent tracked separately. The matmul output is int32 with block_exp — that's MTFP by definition.

**What is normalization actually doing?** It's fitting the mantissa into the designated range (±MANT_MAX) by shifting trits. This is necessary when the mantissa overflows. But for the trivials: ReLU doesn't change magnitude (only zeros negatives). Square can overflow (29524² = 871M > 21.5M). Multiply can overflow (871M × 29524 = 25.7T > 21.5M). So normalization IS needed after square and multiply. But it doesn't have to happen between each op — it can happen once at the end.

**Deferred normalization is native MTFP.** The mantissa is allowed to be larger than MANT_MAX temporarily, with the understanding that it represents mantissa × 3^exponent. The normalization at the end is the same operation — just done once instead of three times. The intermediate values are MTFP with extended mantissa range.

## Hidden assumption: each MTFP op must normalize its output

The mtfp21_mul function normalizes after every call. This is correct for general-purpose arithmetic. But for a known sequence of ops (relu → square → multiply), normalization after relu is wasted (relu doesn't change magnitude), normalization after square is wasted if multiply follows immediately, and only the final normalization matters.

This is the same insight as the batched normalization in the matmul — defer normalization to the end of a known sequence. It worked for the matmul. It'll work for the trivials.

## What would this look like if it were easy?

```
int32 gate_raw = matmul_accumulator;  // from the ternary dot product
int8  gate_exp = block_exp;           // from the block alignment

// ReLU: check sign of the int32. No normalization.
if (gate_raw < 0) { gate_raw = 0; }

// Square: int32 × int32 → int64. Exponent doubles. No normalization.
int64 gate_sq = (int64)gate_raw * gate_raw;
int8  sq_exp = gate_exp * 2;

// Multiply by up: int64 × int32 → int64. Exponents add. No normalization.
int64 result = gate_sq * (int64)up_raw;
int8  res_exp = sq_exp + up_exp;

// NOW normalize to MTFP21. One time.
mtfp21_t out = normalize(result, res_exp);
```

Three operations. Zero intermediate normalizations. The mantissa is int64 (fits the extended range). The exponent is tracked as a simple int8. The final normalization is the only trit-shift loop.

But this requires the matmul to output raw int32 + block_exp instead of MTFP21. Which is... what the matmul ACTUALLY produces before we normalize it. The normalization inside shirley_gemv_mtfp16 is the waste — we normalize to MTFP21, then the trivials would un-normalize (extend) to int64 for the square/multiply, then re-normalize.

## Core insight

**The matmul should output raw int32 + block exponent. The trivials should operate on raw integers with tracked exponents. The normalization to MTFP21 should happen ONCE — at the boundary where MTFP21 is actually needed (the sub_norm rsqrt).**

This is deferred normalization across the entire gate+up → relu → square → multiply pipeline. The data format between the matmul output and the sub_norm input is "extended MTFP" — int32 or int64 mantissa with int8 exponent. Not a new format — just MTFP without the mantissa range constraint.

## Stacking the step-changes

The step-changes aren't mutually exclusive. They stack:

| Change | Improvement | Stacks with |
|--------|-------------|-------------|
| Remove ggml boundary | +12% | Everything |
| Attention multi-threading | +11% | Everything |
| Fused trivials (deferred norm) | ~+10% | Remove ggml |
| int8 matmul (32 lanes) | ~+15% | Needs validation |
| iGPU EXP dispatch | ~+5% at long sequences | Everything |

Stacked: 4.57 × 1.12 × 1.11 × 1.10 × 1.15 ≈ 7.3 tok/s. That's 33% of baseline — up from 21%.

If the int8 matmul doesn't lose quality: 4.57 × 1.12 × 1.11 × 1.10 × 1.15 ≈ 7.3 tok/s. If it does, without it: ~6.3 tok/s.

These are rough estimates. The real numbers need measurement. But the direction is clear: the step-changes are structural (remove waste, fuse ops, multi-thread) not incremental (tune SIMD, adjust parameters).
