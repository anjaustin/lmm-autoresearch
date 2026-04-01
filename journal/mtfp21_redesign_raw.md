# RAW: MTFP21 Kernel Redesign

## Stream of Consciousness

We built a correct number system and then used it wrong. MTFP21 was supposed to bridge the float ocean between ternary matmul islands. Instead we drowned in it — wrapping every per-element operation in a mantissa+exponent struct, normalizing after every multiply, paying 40 instructions for what sign_epi8 does in 1.

The ternary AVX kernel works because it trusts the hardware. Load, sign, add. Three instructions. The hardware does ternary multiplication natively and has since 2004. We knew this. We benchmarked it. We validated it in the matmul. And then for RMSNorm we forgot everything and built a software FPU.

The core mistake was treating MTFP21 as a universal compute format. It's not. It's a representation format — a way to encode values with dynamic range in balanced ternary. The COMPUTE should happen in the format the hardware understands: raw integer vectors via AVX2 for the per-element work, and either a fast scalar path (LUT+NR) or iGPU dispatch for the one transcendental.

RMSNorm is: square all elements, sum them, divide by n, add epsilon, take rsqrt, scale all elements. Five of those six steps are ADD or MUL — primes that the CPU handles natively at 32 elements per cycle. The sixth (rsqrt) is called once and takes 0.2 microseconds even in the unoptimized MTFP21 path. We spent 180 microseconds on the five easy steps and 0.2 on the hard one.

The profiler told us: 47% of time is in mtfp21_mul (5120 calls), 21% in mtfp21_add (2560 calls), 0.1% in rsqrt (1 call). The solution isn't to optimize mtfp21_mul. It's to not call it 5120 times. The per-element squaring is int8 × int8 = int16 — one native instruction, 32 elements per cycle via AVX2. The accumulation is int16 → int32 via madd_epi16 — one native instruction. The scale is int8 × scalar — one multiply-shift-pack per 32 elements.

The iGPU question is about batching. One rsqrt on the CPU costs 0.2 us. One rsqrt on the iGPU costs maybe 0.1 us compute but 1+ us dispatch overhead. For a single call, CPU wins. But a forward pass has 120 RMSNorms (30 layers × 4 norms each). If we batch all 120 rsqrts into one iGPU kernel launch, the amortized dispatch overhead is ~0.01 us per rsqrt, and the iGPU compute is 4 cycles per rsqrt on hardware EXP+LOG. That's the substrate routing Shirley predicts.

But honestly — the iGPU optimization is premature. The per-element integer path is where the 900x speedup lives. The rsqrt is 0.1% of the time. Fixing the 99.9% first, then optimizing the 0.1%, is the right order.

What I'm worried about: overflow. int8 × int8 = up to 127 × 127 = 16129, which fits in int16 (max 32767). Sum of 2560 int16 squares: up to 2560 × 16129 ≈ 41M, which does NOT fit in int32 if we're summing int16 directly. It fits in int32 (max 2.1B) if we widen to int32 during accumulation. So the accumulation needs to be int16 → int32, not int16 → int16.

For the scale multiply: the rsqrt result is a float or fixed-point scalar. Multiplying int8 × float_scalar → float → round → clamp → int8 uses float for one multiplication per element. That's one float multiply per element, which at AVX2 width is 8 floats per cycle (mulps). For 2560 elements, that's 320 cycles ≈ 100 ns. Fast enough.

Or: keep the scale as a fixed-point integer. If scale = rsqrt_mantissa × 3^rsqrt_exp, we can decompose the multiply as: `result = (int8_val * rsqrt_mantissa) >> shift`. This stays entirely in integer but requires careful shift computation. The shift is determined by the rsqrt exponent and the desired output range. Doable, but more engineering.

The pragmatic choice: use float for the single rsqrt and the single scale-multiply-per-element. One float rsqrt costs 5 ns. One vectorized float multiply costs 100 ns. Total float involvement: 105 ns for the parts that need dynamic range, pure integer for the rest. Total RMSNorm: ~130 ns vs 180,000 ns currently. That's 1400x faster.

Wait — but the whole point of MTFP21 was to avoid float. If we use float for rsqrt and scaling, what was the point?

The point was to prove that the arithmetic COULD be done in integer with sufficient precision. We proved that. 102/102 tests. Better than float32 for accumulation. The MTFP21 library is the proof of concept. The production path uses integer where it's fast (squaring, accumulation) and float where it's fast (rsqrt, scaling), because both are native to the CPU.

The Shirley thesis isn't "never use float." It's "route computation to the substrate that handles it best." On this Ryzen, the best substrate for rsqrt is either the FPU (5 ns) or the iGPU (4 cycles hardware). The best substrate for element-wise integer multiply is AVX2 integer. Using MTFP21 for both is using the wrong substrate for both.

MTFP21's role: transport format between heterogeneous compute stages. Representation for values that need balanced ternary + dynamic range. NOT the per-element compute format. The compute format is whatever the hardware does fastest for each prime.

## Open Questions

1. For the scale multiply, is float32 acceptable or does the full-integer path matter for the thesis?
2. Can we structure the iGPU dispatch to batch all 120 rsqrts (one per RMSNorm in the forward pass)?
3. What's the minimum overhead for an iGPU kernel launch on the Ryzen integrated graphics with shared memory?
4. Should we keep the MTFP21 scalar path as a fallback for systems without FPU or iGPU?
5. Can the accumulation step use _mm256_madd_epi16 directly on int8 pairs to avoid explicit widening?

## First Instincts

- Build the AVX2 integer RMSNorm kernel. It's the 99.9% speedup.
- Use float rsqrt for now. Profile it. If it's < 1% of total, it's fine.
- Keep MTFP21 as the proof-of-concept library and as a transport format.
- The iGPU batched rsqrt is a Phase 2 optimization — measure the dispatch overhead first.
- The integer-only scale multiply is a Phase 3 optimization — only if float proves unacceptable.
