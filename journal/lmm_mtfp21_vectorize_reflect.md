# REFLECT: MTFP21 Vectorization — Third Way

## Why three times on Node 2 (exponent hidden in the word):

**Why does IEEE float work in SIMD?** Because the exponent and mantissa are fused into one 32-bit word, and the FPU handles them atomically. One instruction, one cycle.

**Why can't MTFP21 do the same?** Because the mantissa is int32 (needs all 32 bits) and the exponent is separate. There's no room in 32 bits for both. Unless we reduce the mantissa.

**Why do we need 32-bit mantissa?** Because 16 trits need 25.4 bits of integer range (±21.5M). But that's the FULL precision. The operations between matmuls don't need full precision — the matmul packing drops to int8 anyway.

## Hidden assumption: MTFP21 must be one format everywhere

I've been assuming the MTFP21 representation is fixed — 16-trit mantissa, 5-trit exponent, always. But what if the representation ADAPTS based on where in the pipeline we are?

- At matmul boundaries: int8 mantissa (7-8 trits) with MTFP21 scale factor
- Between matmuls (FFN): int16 mantissa (10-11 trits) with shared exponent — block MTFP
- In scalar computations (rsqrt, exp): full MTFP21 (16 trits)
- In the KV cache: whatever precision the geometric structure requires

This isn't three different number systems. It's ONE number system at different precision levels, like float16/float32/float64 are all IEEE at different widths. The exponent semantics are the same. The mantissa width varies by context.

## What would this look like if it were easy?

The FFN between matmuls uses int16 mantissas with a shared block exponent. 16 lanes per AVX2 register. Multiply is `_mm256_mullo_epi16`. The block exponent updates once per vector operation, not per element. ReLU is `_mm256_max_epi16`. Square is `_mm256_mullo_epi16` with exponent doubling. Gate×up is `_mm256_mullo_epi16` with exponent adding.

When we need full precision (rsqrt for RMSNorm), we widen to int32 and use full MTFP21. When we pack for matmul, we narrow to int8. The precision flows through the pipeline like water — wider where it needs to be, narrower where speed matters.

## What assumption might be wrong?

That we need MORE than int16 between matmuls. The FFN operations (ReLU, square, multiply) don't accumulate error — they're pointwise. The error only accumulates across layers through the residual connection. 10-11 trits (int16) might be plenty for 3-5 pointwise operations before the next matmul drops to int8 anyway.

## Core insight:

**Adaptive-width MTFP.** Same base-3 exponent semantics everywhere. Mantissa width adapts to the operation: int8 at the matmul wire, int16 for SIMD bulk, int32 for scalar precision. The exponent is the invariant — the geometric coordinate that persists across width changes. The mantissa is the local detail that can be truncated when speed matters.

This is the third approach. Not block exponent (which forces a shared exponent). Not full per-lane (which keeps per-element exponents). Adaptive width with per-element exponents at the natural word size for each operation.

## Resolved tensions:
- Node 3 vs Node 4: Don't choose log vs linear. Use int16 linear for everything. Multiply is one instruction. Add requires exponent alignment but with only 10-trit mantissas, alignment is a shift + mask, not a loop.
- Node 6 vs precision: 10-11 trits is the right answer for between-matmul operations. Full MTFP21 stays available for scalar ops.
- Node 7 vs Node 4: With per-element exponents (not shared), wide dynamic ranges in attention are handled. Int16 mantissa × int16 mantissa → int32, with exponent sum. The widening from 16→32 is natural SIMD (`_mm256_cvtepi16_epi32` + `_mm256_mullo_epi32`).
