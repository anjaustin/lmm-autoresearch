# RAW: MTFP21 Vectorization — Is There a Third Way?

We have two approaches to vectorize MTFP21: block exponent (simple, lossy at edges) and full per-lane (precise, slower). Both are valid. Both have clear trade-offs. But I'm being asked whether there's something we're not seeing.

Let me dump everything I think I know and look for what I'm assuming.

MTFP21 is mantissa × 3^exponent. The mantissa is int32. The exponent is int8. The six primes operate on these. The hardware is AVX2 — 256-bit registers, 8×int32 or 32×int8 or 4×int64.

What makes MTFP21 hard to vectorize is that each element has its own exponent. Operations between elements with different exponents require alignment (for add) or produce results with new exponents (for multiply). This per-element bookkeeping is inherently serial — or is it?

What if the exponent isn't a separate thing to track? What if it's part of the value itself?

In IEEE float32, the exponent IS part of the value — it's bits 23-30 of the 32-bit word. The hardware doesn't "track" the exponent separately. It's baked into the representation. When you multiply two floats, the hardware adds the exponents as part of the single multiply instruction. There's no separate exponent step.

MTFP21 has a SEPARATE exponent because the mantissa is int32 and the exponent is int8 — they're two fields in a struct. This separation is what creates the vectorization problem. If they were fused into a single machine word, the SIMD lane would handle both at once.

But wait — can we fuse them? The mantissa is int32 (up to 21.5M). The exponent is int8 (-121 to +121). Together that's 40 bits. Doesn't fit in int32. Fits in int64 — but int64 SIMD gives us only 4 lanes.

What if we don't need the full int32 mantissa for SIMD operations? The full MTFP21 mantissa has 16 trits of precision (25.4 bits). But for the SIMD hot path (bulk element-wise operations between matmuls), do we need all 16 trits?

The int8 matmul wire format already drops to ~7 trits (127 levels). The MTFP21 operations between matmuls (ReLU, square, multiply, normalization) might not need full 16-trit precision either — they're intermediate steps that feed into another matmul quantization.

What if we use a REDUCED precision MTFP for the SIMD path? Something like:
- 10-trit mantissa: fits in int16 (range ±29524)
- 5-trit exponent: fits in int8 (range ±121)
- Total: 24 bits — packs into int32 with room to spare

With int16 mantissas, we get 16 lanes per AVX2 register. Multiply is `_mm256_mullo_epi16` — ONE instruction, 16 lanes. The exponent fits in the upper 8 bits of each int32 lane, or in a separate int8 register.

Actually — pack the mantissa and exponent together:
- Lower 16 bits: mantissa (int16, 10 trits)
- Upper 16 bits: exponent (int16, but only uses ±121)
- Total: one int32 per value, 8 lanes per AVX2 register

Multiply: extract mantissas (mask + shift), multiply (int16 × int16 → int32 via `_mm256_madd_epi16`), add exponents (extract + add), repack. Three instructions for the mantissa multiply, two for the exponent. Five instructions per 8 elements.

But the precision drops from 25.4 bits (16 trits) to 15.8 bits (10 trits). Is that enough?

The key insight from our work: MTFP21 between matmuls was needed because int8 quantization lost 1+ PPL. But MTFP21's 25.4-bit precision was overkill — we never measured how much precision we actually NEED between matmuls. It's somewhere between 8 bits (too lossy) and 25.4 bits (validated). Where's the floor?

10 trits = 59049 levels = 15.8 bits. That's less than float16's 11-bit mantissa (which is widely used for inference without quality loss). So 10 trits might be too few. 

12 trits = 531441 levels = 19.0 bits. Better than float16. Fits in int32 with the exponent.

13 trits = 1594323 levels = 20.6 bits. Still fits in int32 (21 bits for mantissa + 8 bits for exponent = 29 bits < 32).

What about this: pack a 13-trit mantissa (int32, range ±797161) and a 5-trit exponent into a SINGLE int32:
- Bits 0-20: mantissa (21 bits, holds ±797161)
- Bits 21-28: exponent (8 bits, holds ±121)
- Bits 29-31: unused (or sign extension)

This is a FUSED MTFP representation — one int32 per value, 8 lanes per AVX2 register. The exponent and mantissa live in the same word. Operations extract, compute, repack.

But the extract/repack overhead might negate the lane-width gain. Let me think about what operations actually need...

For ReLU: `max(x, 0)`. If the fused representation preserves sign in the mantissa, ReLU is just `_mm256_max_epi32(x, zero)` — IF negative values have negative int32 representations. With mantissa in the low bits and exponent in the high bits, negative mantissa = negative low bits, positive exponent in high bits → the int32 value is positive even when the MTFP value is negative. ReLU breaks.

For square: `x * x`. Mantissa squared, exponent doubled. The fused multiply is not a simple int32 multiply — it's mantissa × mantissa with exponent + exponent. Can't use `_mm256_mullo_epi32` directly.

OK so the fused representation doesn't give us single-instruction operations. The packing just saves memory, not compute.

What am I actually trying to solve? The bottleneck is per-element MTFP21 arithmetic in the FFN and attention. The operations are: multiply, add, compare, negate. The expensive part of multiply is the int64 product + normalization. The expensive part of add is exponent alignment.

What if normalization wasn't needed? What if we let mantissas grow and only normalize at specific points (before matmul packing, before overflow)?

Deferred normalization: do multiplies as int64 products, accumulate exponents, but don't normalize until a checkpoint. Between two matmuls, there are ~5 operations (ReLU, square, multiply, norm, quantize). If we defer normalization, each multiply just does int64 × int64 → int128... that's worse.

What if we use LOGARITHMIC representation instead? In log space, multiplication is addition. The entire MTFP21 multiply (int64 product + normalization) becomes int32 addition in log space. That's ONE instruction.

Log-MTFP: value = 3^(fixed_point_log_value). The entire value is one int32 representing the log base 3 of the number. Multiplication: add the log values. Division: subtract. These are single SIMD instructions.

Addition in log space is hard: log(a + b) = log(a) + log(1 + 3^(log(b) - log(a))). That requires a log-add LUT. But ReLU doesn't need addition — it's max(x, 0) = max of the log values. Square doesn't need addition — it's 2× the log value. The element-wise multiply gate²×up is just adding log values.

The operations that NEED addition: residual connections and attention accumulation (Σ scores × V). These are the only places where log-space is expensive.

Wait. The FFN between matmuls:
1. ReLU(gate): max(x, 0) — works in log space (compare to -∞)
2. Square(ReLU): 2× — works in log space (shift)
3. gate² × up: add — works in log space (add)
4. RMSNorm: needs addition (sum of squares) — expensive in log space

And attention:
5. Q · K: dot product — needs addition — expensive in log space
6. Softmax: exp() — trivial in log space (it's already log!)
7. Σ scores × V: needs addition — expensive

So log-space wins for multiply-heavy chains but loses for addition-heavy chains. The FFN is multiply-heavy (ReLU, square, gate×up). The attention is addition-heavy (dot products, weighted sums).

Three open questions:
1. Is there a representation that makes BOTH multiply and add fast in SIMD?
2. Can the block exponent approach be made adaptive — different block sizes for tight vs wide distributions?
3. Is the 10-trit reduced MTFP worth exploring if it gives 16 lanes?
