# NODES: MTFP21 Vectorization — Third Way

## Node 1: The bottleneck is per-element normalization
MTFP21 multiply produces a product that may overflow the mantissa range. Normalization (divide by 3^n) is iterative and data-dependent. This is what makes it serial.

## Node 2: IEEE float hides the exponent in the word
Float32 doesn't "track" exponents separately. The hardware handles exponent arithmetic as part of the multiply instruction. MTFP21's two-field struct is the source of the vectorization problem.

## Node 3: Log-space eliminates multiply normalization
In log representation, multiply = add. No normalization. One SIMD instruction. But add becomes expensive (requires log-add LUT).

## Node 4: The FFN is multiply-heavy, attention is add-heavy
FFN between matmuls: ReLU (compare), square (2×), gate×up (add in log). All cheap in log.
Attention: dot products (many adds), weighted sums (many adds). Expensive in log.

## Node 5: Block exponent is block floating point
This is a known technique (BFP). Microsoft's MSFP, Google's BFloat, etc. The innovation would be making it MTFP-native — base-3 block exponent instead of base-2.

## Node 6: Reduced precision might be sufficient
We validated that int8 (8 bits) loses 1+ PPL and MTFP21 (25.4 bits) loses nothing. The actual precision floor is somewhere between. 12-13 trits (19-21 bits) might be the sweet spot — more than float16 (11-bit mantissa), fits in int32 with the exponent.

## Node 7: The geometric interpretation suggests something
Numbers are positions. The exponent is a coordinate. In a geometric space, nearby points have similar coordinates. After RMSNorm, values ARE nearby — their exponents cluster. The block exponent works because the geometry is locally flat after normalization.

## Node 8: Hybrid log/linear
Log for multiplies (FFN trivials), linear for adds (attention accumulators). Switch representation at the operation boundary. The conversion is one LUT lookup per element per switch.

## Node 9: The matmul already packs to int8
At every matmul boundary, we convert to int8 anyway. The SIMD path only needs to be fast BETWEEN matmuls. That's 3-5 operations in the FFN and the dot product + softmax + weighted sum in attention.

## Tensions:
- Node 3 vs Node 4: Log-space and linear-space each win for different operations. Neither wins everywhere.
- Node 6 vs precision: Reducing precision risks repeating the int8 mistake. But 12 trits >> 8 bits.
- Node 7 vs Node 4: Geometric locality (block exponent works after norm) breaks down in attention where values span wide ranges.
- Node 8 vs simplicity: Hybrid representations add conversion overhead and implementation complexity.
