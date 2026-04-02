# Adaptive-Width MTFP

## The Family

MTFP is not one number system. It is a family of representations that share a single invariant: the base-3 exponent.

```
value = mantissa × 3^exponent
```

The exponent is the geometric coordinate — the position in base-3 space. The mantissa is the local detail at that position. The width of the mantissa determines precision. The exponent is always the same.

| Format | Mantissa | Precision | AVX2 Lanes | Use |
|--------|----------|-----------|------------|-----|
| MTFP8  | int8 (±121) | 4.9 trits, 7.8 bits | 32 | Memory-constrained ops |
| MTFP16 | int16 (±29524) | 10 trits, 15.8 bits | 16 | **Matmul wire (sign_epi16)** |
| MTFP21 | int32 (±21523360) | 16 trits, 25.4 bits | 8 | Precision ops (rsqrt, exp) |

All three share the 5-trit exponent (int8, range ±121). Converting between them is mantissa truncation (wider → narrower) or mantissa extension (narrower → wider). The exponent does not change.

## Why Adaptive Width

A single format forces a choice: precision or parallelism. MTFP21 has 25.4-bit precision but only 8 SIMD lanes (int32). MTFP8 has 32 SIMD lanes but only 7.8-bit precision — worse than the int8 quantization that lost 1+ PPL.

Adaptive width eliminates the choice. Each operation gets the width it needs:

- **RMSNorm rsqrt:** needs full precision for the Newton-Raphson iterations → MTFP21
- **Softmax exp:** needs full precision for the LUT + interpolation → MTFP21
- **RoPE:** MUL + ADD on precomputed constants → MTFP21 (could be MTFP16)
- **Ternary matmul:** sign_epi16 × 2-bit weights → MTFP16 (16 lanes)
- **ReLU, square, element-wise multiply:** pointwise ops between matmuls → MTFP21

The precision floor for between-matmul ops is MTFP21 (25.4 bits). The matmul wire is MTFP16 (15.8 bits). The matmul wire precision is sufficient because:
1. RMSNorm normalizes the data to a tight range before each matmul
2. The ternary weights are only {-1, 0, +1} — the activation precision is the limiting factor
3. 15.8 bits exceeds float16's 11-bit mantissa, which is widely used for inference
4. The block exponent alignment loses nothing on post-normalization data (zero values crushed = 0)

## The Production Matmul: sign_epi16

The ternary multiply instruction for MTFP16 is `_mm256_sign_epi16`:

```
result = sign_epi16(activation_mantissa, ternary_weight)

  weight = +1: result = activation    (pass through)
  weight =  0: result = 0             (zero)
  weight = -1: result = -activation   (negate)
```

Same semantics as sign_epi8 but operating on int16 mantissas. 16 lanes per AVX2 register.

The full matmul kernel (`shirley_mtfp16_matmul.h`):
1. Load 32 packed bytes → unpack to 4 × 32 int8 ternary values
2. Sign-extend int8 → int16: `_mm256_cvtepi8_epi16`
3. Ternary multiply: `_mm256_sign_epi16(activation, weight)`
4. Pair-wise horizontal add: `_mm256_madd_epi16(product, ones)` → int32
5. Accumulate int32
6. Horizontal sum → int32 result with block exponent → MTFP21

Zero float conversion. The int16 mantissa comes from block-aligned MTFP21 values. The int32 result lifts directly to MTFP21 with the block exponent.

## Block Exponent Alignment

Before the matmul, MTFP21 values are converted to a block of int16 mantissas sharing a single exponent:

1. Convert each MTFP21 to MTFP16 (truncate mantissa to ±29524)
2. Find the maximum exponent across all elements
3. Align each mantissa: trit-shift right by (max_exp - element_exp) positions
4. Result: int16 array + one int8 block exponent

This works because RMSNorm output has a tight dynamic range — all exponents cluster near the same value. The alignment loses almost nothing (validated: 0 values crushed to zero on post-normalization data, max error 0.5% across 1000 random vectors).

## Width Transitions

```
MTFP21 → MTFP16:  trit_rshift mantissa until it fits int16 range, increment exponent
MTFP16 → MTFP21:  sign-extend int16 → int32, normalize (trit_lshift to maximize precision)
MTFP21 → block:   MTFP21 → MTFP16 → align to shared exponent → int16 array
block → MTFP21:   int32 accumulator + block exponent → normalize
```

Each transition preserves the exponent semantics. The geometric coordinate is invariant.

## Validation

`test_mtfp16_matmul.c` — 6/6 tests pass:
- MTFP16 conversion round-trip: max error 5.08e-05
- Small dot product (n=8): all 3 paths match float reference to 7.6e-05
- Realistic dot product (n=2560): 3.4e-05 relative error, 0 values crushed
- Statistical (1000 trials, n=2560): avg error 8e-05, max 0.5%

In the full pipeline: "The capital of France is Paris, and the capital of Germany is Berlin."

## Origin

Adaptive-width MTFP discovered via Lincoln Manifold Method, April 2, 2026. The RAW phase explored block exponent, per-lane SIMD, fused representations, log-space, and reduced precision. The REFLECT phase uncovered the hidden assumption: "MTFP21 must be one format everywhere." The SYNTHESIZE phase produced the family: same exponent, different mantissa widths, precision adapts to the operation.
