# BitNet End-to-End Ternary Conversion Plan

## The Problem

BitNet b1.58 stores weights in ternary {-1, 0, +1} but computes in float32. The inference pipeline unpacks ternary weights into a float32 dot product at every layer, then float32 activations flow through RMSNorm, SiLU, softmax — all float32 — before the next layer's ternary weights are unpacked again into another float32 dot product.

The model thinks in ternary but computes in float. The conversion happens 6 times per layer (6 matmuls), plus 4 transcendental operations (2× RMSNorm, 1× softmax, 1× SiLU). For a 28-layer model, that's 168 float32 matmuls and 112 transcendental operations that could be ternary.

## The Goal

Replace float32 operations with ternary integer operations one at a time, measuring output quality after each swap. When quality breaks, we know exactly which operation caused it and can study why.

## Two Paths

### Path A: 5-Trit Activations
Quantize activations to 5 balanced ternary trits (243 states, ~25 bits) between operations. Based on today's finding that 5-trit quantization acts as a denoiser — the precision boundary removes noise rather than signal.

**Advantage:** Simple, fast (sign_epi8 throughout), proven denoising effect on MNIST.
**Risk:** Denoising compound effect across 28 layers. What helps in one layer might destroy signal by layer 28. Unknown — must be measured.

### Path B: MTFP21 Activations
Represent activations as MTFP21 (16-trit mantissa + 5-trit exponent). Exceeds float32 precision (25.4 bits vs 24 bits). All arithmetic stays in AVX2 integer pipeline via sign_epi8.

**Advantage:** Higher precision than float32, so quality should be preserved or improved.
**Risk:** MTFP21 addition is complex (exponent alignment, ternary right-shift rounding). More engineering. Untested arithmetic.

### Which first?
Path A is simpler and answers the most interesting question: does ternary quantization between layers help or hurt? If it helps (or is neutral), we don't need MTFP21 for activations at all. If it hurts, we know the precision floor and can design MTFP21 accordingly.

Start with Path A. Fall back to Path B if needed.

## Method: One Operation at a Time

### Prerequisites
1. BitNet b1.58-2B model loaded and running on the Ryzen
2. Baseline perplexity measured on a fixed eval set (WikiText-2 or similar)
3. Baseline token output on a fixed set of prompts (for qualitative comparison)
4. Build system configured so we can swap individual operations

### Conversion Order (easiest and most impactful first)

**Phase 1: Matmul kernel swap**
Replace `_mm256_maddubs_epi16` (unsigned multiply-add with {0,1,2} encoding) with `_mm256_sign_epi8` (ternary multiply with {-1,0,+1} encoding). This is the operation we benchmarked at 6.87x faster. The weights are already ternary — we're just changing how they participate in the dot product.

```
Step 1.1: Replace ONE matmul (e.g., first QKV projection in layer 0)
          Measure: perplexity, token output
Step 1.2: Replace ALL QKV projections (all layers)
          Measure: perplexity, token output
Step 1.3: Replace ALL matmuls (QKV + output + FFN gate/up/down)
          Measure: perplexity, token output
```

If perplexity is unchanged after 1.3, the matmul swap is validated. This alone gives the 6.87x speedup on the dominant compute operation.

**Phase 2: Activation quantization**
After each matmul, quantize the float32 output to 5-trit balanced ternary before passing to the next operation. This is the big question — does inter-layer quantization help, hurt, or make no difference?

```
Step 2.1: Quantize activations after ONE matmul (first QKV, layer 0)
          Keep everything else float32
          Measure: perplexity
Step 2.2: Quantize after ALL matmuls in layer 0
          Measure: perplexity
Step 2.3: Quantize after ALL matmuls in ALL layers
          Measure: perplexity
          This is the compound effect test
```

If perplexity degrades at step 2.3 but not 2.1, the compounding is the issue. Try with more trits (6, 7, 8) to find the minimum precision that survives 28 layers.

**Phase 3: Residual connections**
Convert residual additions from float32 to ternary integer ADD.

```
Step 3.1: Convert residual adds in layer 0
          Measure: perplexity
Step 3.2: Convert ALL residual adds
          Measure: perplexity
```

This should be safe — ADD is exact in integer if the values fit. The question is whether the accumulated integer values stay within int8/int16/int32 bounds.

**Phase 4: Element-wise multiply (FFN gate × up)**
Convert from float32 multiply to sign_epi8 ternary multiply.

```
Step 4.1: Convert in layer 0
          Measure: perplexity
Step 4.2: Convert all layers
          Measure: perplexity
```

**Phase 5: RMSNorm**
This is the first transcendental domain crossing. RMSNorm requires sqrt (= EXP + LOG).

Options:
a) Dispatch EXP/LOG to iGPU (V_EXP_F32, V_LOG_F32)
b) Approximate with frozen ternary lookup table
c) Use MTFP21 for this one operation

```
Step 5.1: Replace RMSNorm in layer 0 with option (a), (b), or (c)
          Measure: perplexity
Step 5.2: Replace ALL RMSNorms
          Measure: perplexity
```

**Phase 6: SiLU activation**
SiLU = x × sigmoid(x) = x × exp(x) / (1 + exp(x)). Same EXP domain crossing as RMSNorm.

```
Step 6.1: Replace SiLU in layer 0
          Measure: perplexity
Step 6.2: Replace ALL SiLUs
          Measure: perplexity
```

**Phase 7: Softmax (attention)**
Softmax = exp(x_i) / Σexp(x_j). EXP domain crossing plus normalization.

```
Step 7.1: Replace softmax in layer 0
          Measure: perplexity
Step 7.2: Replace ALL softmax
          Measure: perplexity
```

**Phase 8: Embedding + LM head**
Convert embedding table to multi-trit storage. Convert final LM head matmul.

```
Step 8.1: Convert embedding
Step 8.2: Convert LM head
          Measure: perplexity (this is the last piece)
```

### After Each Step

Record in results log:
- Which operation was converted
- Which layers were converted
- Perplexity on eval set
- Sample outputs on fixed prompts
- Any numerical issues (NaN, Inf, overflow)
- Throughput measurement (tokens/sec)

When quality breaks, STOP and document:
- Which operation broke it
- At which layer depth
- What the failure mode looks like (gradual degradation? sudden collapse? specific token patterns?)
- Hypothesis for why
- What the failure tells us about the precision requirements

## Expected Outcome

The matmul swap (Phase 1) should be clean — we're just changing the encoding from {0,1,2} to {-1,0,+1} and using a faster instruction. No precision change.

The activation quantization (Phase 2) is the big unknown. If 5-trit survives 28 layers, the entire linear compute path goes ternary integer. If it doesn't, we'll know the precision floor and can use MTFP21 or a higher trit count.

The transcendentals (Phases 5-7) are the hardest. These are where the iGPU earns its place — or where frozen shape approximation proves itself.

## Connection to Today's Work

- 6.87x dot product speedup → Phase 1 matmul swap
- 5-trit quantization as denoising → Phase 2 activation quantization
- Six primes + hardware mapping → Phases 5-7 transcendental dispatch
- MTFP21 → fallback if 5-trit activations don't survive layered compounding
- Failure as structural scaffolding → every break point is a finding

## What We Need Before Starting

- [ ] BitNet b1.58-2B model file on this machine
- [ ] BitNet.cpp building and running baseline inference
- [ ] Fixed eval set for perplexity measurement
- [ ] Fixed prompt set for qualitative comparison
- [ ] Build system that lets us swap individual operations
- [ ] Baseline measurements recorded
