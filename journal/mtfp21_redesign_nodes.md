# NODES: MTFP21 Kernel Redesign

## Extracted Nodes

### Node 1: The per-element work should use AVX2 integer, not MTFP21
sign_epi8 does ternary multiply at 32 elements/cycle. mullo_epi16 does integer square at 16 elements/cycle. madd_epi16 does widening pair-add at 16 elements/cycle. These are the native instructions for the per-element work in RMSNorm. MTFP21 wraps each of these in a normalize-after-every-op loop that costs 40x per element.

### Node 2: MTFP21 is a transport/storage format, not a compute format
MTFP21's value is: balanced ternary representation with dynamic range, exceeding float32 precision, fully expressible in integer. This matters for storing weight scales, normalization parameters, and accumulated statistics — values computed once and used many times. It does NOT matter for per-element squaring and scaling, which are bulk operations on vectors.

### Node 3: The rsqrt is solved and cheap — 0.1% of pipeline time
The LUT+NR rsqrt takes 0.2 us. Float rsqrt takes 5 ns. Either way, it's negligible. The rsqrt was the CORRECTNESS challenge, not the PERFORMANCE challenge. We solved it. Time to move on.

### Node 4: Overflow bounds are tractable for integer RMSNorm
- int8 × int8 → int16: max 127² = 16129, fits in int16 (max 32767)
- Sum of 2560 int16² → need int32: max 2560 × 16129 = 41M, fits in int32 (max 2.1B)
- Actually: for 5-trit inputs (range ±80), max square is 6400, sum is 16.4M. Comfortable.
- Scale multiply: int8 × float_scalar → float → round → int8. Standard quantization path.

### Node 5: The iGPU wins for BATCHED transcendentals, not single calls
One rsqrt on CPU: 0.2 us (LUT+NR) or 5 ns (float sqrtf).
One rsqrt on iGPU: ~1 ns compute but ~1-10 us dispatch overhead.
120 rsqrts batched on iGPU: ~120 ns compute + ~1-10 us dispatch = ~10 us amortized.
120 rsqrts on CPU: 120 × 0.2 us = 24 us (LUT+NR) or 120 × 5 ns = 0.6 us (float).

The iGPU wins over LUT+NR for batched transcendentals if dispatch overhead < 14 us.
The iGPU loses to float sqrtf unless dispatch overhead < 0.6 us.
This needs measurement, not speculation.

**Tension with Node 3:** If float rsqrt is 5 ns and the total pipeline is ~130 ns, the transcendental is 4% of runtime. The iGPU can only improve that 4%. The ROI of iGPU integration is bounded by Amdahl's law.

### Node 6: The AVX2 integer RMSNorm kernel has a natural structure
```
Phase 1: Square — _mm256_mullo_epi16 on widened int8 pairs
Phase 2: Accumulate — _mm256_madd_epi16 to pair-add, then horizontal sum
Phase 3: Scalar rsqrt — one call (float, LUT+NR, or iGPU)
Phase 4: Scale — _mm256_mullo_epi16(int8, scale) with rounding and pack
```
Each phase maps to 1-2 AVX2 intrinsics. The kernel structure mirrors bench_dot.c.

### Node 7: Using float for rsqrt and scaling is pragmatic, not a betrayal
The Shirley thesis: route computation to the substrate that handles it best.
On this Ryzen, the FPU IS one of the substrates. Using sqrtf for rsqrt and
mulps for vectorized scaling IS routing to the best substrate — the FPU handles
these operations in 1 cycle per element, purpose-built in silicon.

The thesis isn't "never use float." It's "don't use float where integer is native."
sign_epi8 for ternary multiply: integer is native, float would be wrong.
sqrtf for rsqrt: float is native, integer is a software emulation.

**Tension with MTFP21 motivation:** We built MTFP21 to prove float could be eliminated.
We proved it (102/102 tests). The proof of concept is the thesis statement.
The production path uses each substrate where it's fastest.

### Node 8: The existing BitNet pipeline already does most of this
Looking at the actual BitNet code:
- `quantize_row_i8_s`: quantizes float activations to int8 (range ±SHIRLEY_ACT_RANGE)
- Matmul kernel: int8 × 2-bit ternary weights via AVX2
- Post-processing: `result / act_scale * weight_scale` (float)
- RMSNorm: `ggml_compute_forward_rms_norm_f32` — pure float32

The only piece we need to ADD is an integer RMSNorm that takes int8 input
(from the matmul output, after quantization) and produces int8 output
(for the next matmul's activation quantization). The squaring, accumulation,
and scaling are new AVX2 code. The rsqrt can reuse the existing float path
or our LUT+NR.

### Node 9: The architecture is three layers, not one
```
Layer 1: AVX2 integer kernels — per-element bulk operations (sign_epi8, mullo, madd)
Layer 2: Scalar arithmetic — rsqrt, division, scale computation (float or MTFP21)  
Layer 3: Transport/storage — MTFP21 for values that need ternary + dynamic range
```
MTFP21 sits at Layer 3. We were trying to use it at Layer 1. The kernel redesign
puts each layer in its place.

## Key Tensions

1. **Pragmatism vs. purity** (Node 7 vs. MTFP21 thesis): Using float for rsqrt is faster but breaks the "all-integer" narrative. Resolution: the narrative was wrong. The thesis is substrate routing, not float avoidance.

2. **iGPU ROI vs. complexity** (Node 5): Integrating the iGPU adds significant engineering for ~4% of runtime improvement. Resolution: measure dispatch overhead first. If it's > 0.6 us, the float CPU path is faster and the iGPU isn't worth it for rsqrt alone.

3. **MTFP21's role** (Node 2 vs. original design): We designed MTFP21 as the universal compute format. It's actually a transport format. Resolution: this isn't a failure — it's a refinement. The proof of concept validated the arithmetic. The production design uses it where it belongs.
