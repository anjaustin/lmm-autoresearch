# REFLECT: MTFP21 Kernel Redesign

## The Core Insight

**The computation substrate and the representation format are different decisions.**

We conflated them. MTFP21 is a representation: balanced ternary with dynamic range. AVX2 integer is a substrate: hardware that executes ADD, MUL, MAX at 32 elements per cycle. The ternary dot product kernel keeps these separate — it uses int8 as the representation and sign_epi8 as the substrate. We got the best of both.

Then for RMSNorm we merged them. MTFP21 became both the representation AND the compute path. Every per-element operation paid the normalization tax of the representation when all it needed was the speed of the substrate.

The fix separates them again:
- **Substrate for per-element work:** AVX2 integer (mullo_epi16, madd_epi16, sign_epi8)
- **Substrate for transcendentals:** FPU (sqrtf) or iGPU (V_EXP_F32, V_LOG_F32) — whichever is faster
- **Representation for transport:** int8 for activations, MTFP21 for scalar parameters

## Resolving the Tensions

### Pragmatism vs. purity
The Shirley thesis says "route to the best substrate." The FPU is a substrate. It handles rsqrt in 1 cycle. The integer LUT+NR does it in ~60 cycles (0.2 us / ~3 ns per cycle). Using the FPU for rsqrt IS substrate routing. Using integer for rsqrt is substrate MIS-routing — forcing an operation onto hardware that doesn't natively support it.

MTFP21 proved integer rsqrt is POSSIBLE. The production path uses what's FASTEST. Both are true. Both matter.

### iGPU ROI
The numbers: float rsqrt is 5 ns per call. 120 calls per forward pass = 600 ns total. The entire RMSNorm pipeline (after the AVX2 redesign) will be ~130 ns per call, so 120 calls = 15.6 us. The rsqrt is 4% of that. Even if the iGPU does rsqrt in 0 ns, the maximum speedup is 4%.

The iGPU's value isn't in RMSNorm rsqrt. It's in the attention softmax (EXP of n² values, where n is the sequence length). For ctx=512 with 20 attention heads, that's 20 × 512 × 512 = 5.2M EXP operations per layer. THAT is where the iGPU earns its place — not in 120 scalar rsqrts.

For this kernel: use float rsqrt on CPU. Revisit iGPU for softmax.

### MTFP21's role
MTFP21 is not dead. It's repositioned:
1. **Proof of concept:** 102/102 tests prove integer-only inference arithmetic works at 25.4-bit precision. This is a publishable result.
2. **Transport format:** Weight scales, normalization gammas, accumulated statistics can be stored as MTFP21 for ternary-native serialization.
3. **iGPU interface:** When we do integrate the iGPU for softmax, the values passed to the iGPU can be in MTFP21 format (ternary native) rather than IEEE 754.
4. **Fallback compute:** On hardware without FPU (embedded, FPGA), MTFP21 provides the integer-only path.

## Hidden Assumptions Challenged

### "Avoiding float is the goal"
No. Efficient ternary computation is the goal. Float is a tool. sign_epi8 is a tool. The iGPU is a tool. Use each where it's native.

### "MTFP21 should be faster because it has more precision per bit"
Precision per bit is a storage metric, not a compute metric. MTFP21 has excellent information density (33.3 bits in 21 trits). But computing with it requires normalization after every operation — a cost that doesn't exist for fixed-point or native float.

### "The kernel needs to be all one thing"
No. The kernel is a pipeline. Each STAGE uses the best substrate for its operation:
- Squaring: AVX2 int (native)
- Accumulation: AVX2 int (native)
- rsqrt: FPU float (native) or iGPU (native for transcendentals)
- Scaling: AVX2 int or float (both native, both fast)

The pipeline crosses substrate boundaries. That's not a bug — it's the Shirley architecture working as designed.

## What Would This Look Like If It Were Easy?

```c
void ternary_rmsnorm_avx2(int8_t *dst, const int8_t *src, int n, float eps) {
    // Phase 1+2: sum of squares in AVX2 integer
    int32_t sum_sq = 0;
    for (i = 0; i + 32 <= n; i += 32) {
        __m256i v = _mm256_loadu_si256(src + i);
        // Widen to int16, square, pair-sum to int32, accumulate
        ...
    }
    
    // Phase 3: one rsqrt
    float scale = 1.0f / sqrtf((float)sum_sq / n + eps);
    
    // Phase 4: scale all elements
    // Convert scale to fixed-point, multiply int8 × fixed, round, pack
    ...
}
```

50 lines of AVX2 intrinsics. ~130 ns per call. Native on the hardware we have.

## The Structure Beneath the Content

The redesign pattern is: **use the representation that matches the hardware at each stage.**

| Stage | Data | Hardware | Representation |
|-------|------|----------|---------------|
| Matmul | 2560 × 2560 | AVX2 sign_epi8 | int8 trits + 2-bit packed weights |
| Square | 2560 elements | AVX2 mullo_epi16 | int8 → int16 |
| Accumulate | 2560 → 1 | AVX2 madd_epi16 | int16 → int32 |
| rsqrt | 1 scalar | FPU sqrtf | float32 |
| Scale | 2560 elements | AVX2 mullo_epi16 | int8 × fixed-point |
| Output | 2560 elements | — | int8 (for next matmul) |

MTFP21 doesn't appear in this table because none of these stages need dynamic-range balanced ternary. They need raw integers (for bulk work) and one float (for the transcendental). MTFP21 lives in the spaces BETWEEN stages — storing the scale factors, the normalization parameters, the metadata that travels with the integer vectors.
