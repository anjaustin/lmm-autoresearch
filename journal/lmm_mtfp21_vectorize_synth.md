# SYNTHESIS: Adaptive-Width MTFP — The Third Approach

## The Insight

MTFP isn't one format. It's a family. The exponent (the geometric coordinate) is invariant. The mantissa (the local detail) adapts to the operation.

```
MTFP8:  int8  mantissa ×  3^exp  — matmul wire format (sign_epi8, 32 lanes)
MTFP16: int16 mantissa ×  3^exp  — SIMD bulk compute (mullo_epi16, 16 lanes)
MTFP21: int32 mantissa ×  3^exp  — scalar precision (rsqrt, exp, 1 lane)
```

Same number system. Same base-3 exponent. Same geometric interpretation. Different precision for different contexts. Like float16/32/64 are all IEEE — MTFP8/16/21 are all MTFP.

## Why It Works

**Width transitions are cheap.** MTFP16→MTFP21: sign-extend int16→int32, keep exponent. One SIMD instruction (`_mm256_cvtepi16_epi32`). MTFP21→MTFP16: truncate int32→int16 with rounding, keep exponent. MTFP16→MTFP8: same as current pack_for_matmul.

**Per-element exponents survive at int16.** Each MTFP16 value has its own exponent — no shared block exponent, no outlier crushing. The exponent is stored in a separate int8 array alongside the int16 mantissa array. Two arrays, one exponent per element.

**16 lanes for the bulk compute.** The FFN between matmuls: ReLU (`max_epi16`), square (`mullo_epi16` → int32 product, truncate back), gate×up (`mullo_epi16` → int32, truncate). The exponent arrays update with simple int8 adds. 16 elements per cycle.

**Precision is sufficient.** 10 trits = 59049 levels = 15.8 bits of precision. More than float16's 11-bit mantissa. The FFN does 3-5 pointwise operations before the next matmul drops to int8 anyway. The error from 10-trit intermediates is bounded and doesn't accumulate across layers (the matmul packing resets the precision floor each time).

## The Pipeline

```
Embedding (float32) → MTFP21 conversion

Per layer:
  MTFP21 (scalar): RMSNorm — needs rsqrt, full precision
  MTFP21 → MTFP8: pack for matmul wire
  Matmul: int8 × ternary → int32 → MTFP21 (lift result)
  MTFP21 → MTFP16: narrow for SIMD bulk

  FFN bulk (MTFP16, 16 lanes):
    ReLU: _mm256_max_epi16(x, zero)
    Square: _mm256_mullo_epi16(x, x) → handle overflow to int32
    gate × up: _mm256_mullo_epi16 → handle overflow
    Exponent updates: _mm256_add_epi8 on exponent arrays

  MTFP16 → MTFP21: widen for sub-norm rsqrt
  MTFP21 (scalar): RMSNorm
  MTFP21 → MTFP8: pack for down matmul
  Matmul → MTFP21 → add residual

  Attention (MTFP16 for dot products):
    Q·K: 16-lane multiply + horizontal add (needs int32 accumulator)
    Softmax: MTFP21 scalar (exp, one per position)
    Σ scores×V: 16-lane multiply + accumulate

Output:
  MTFP21 → float32 (LM head boundary)
```

## What to Build

1. **MTFP16 type:** `{int16_t mantissa; int8_t exponent;}` — 3 bytes per value (packed), or separate mantissa and exponent arrays for SIMD alignment.

2. **Width converters:** `mtfp21_to_mtfp16()`, `mtfp16_to_mtfp21()`, `mtfp16_to_mtfp8()`. These are truncation/extension + exponent preservation.

3. **MTFP16 SIMD kernels:** `shirley_relu_mtfp16()`, `shirley_square_mtfp16()`, `shirley_mul_mtfp16()`. Each operates on parallel int16 mantissa arrays with int8 exponent arrays.

4. **Integrate into shirley_ffn.cpp:** Replace scalar MTFP21 loops with MTFP16 SIMD for the bulk operations. Keep MTFP21 for RMSNorm scalar path.

## Why This Is the Third Approach

**Approach 1 (block exponent):** Shared exponent, simple, lossy at edges.
**Approach 2 (full per-lane):** Per-element exponent, precise, slow (int64).
**Approach 3 (adaptive width):** Per-element exponent, sufficient precision, fast (int16 lanes). Gets the SIMD width of approach 1 with the per-element exponents of approach 2 by reducing mantissa precision to what the pipeline actually needs.

The geometric coordinate (exponent) is preserved per-element. The local detail (mantissa) is at the precision the operation requires. The representation adapts. The geometry persists.
