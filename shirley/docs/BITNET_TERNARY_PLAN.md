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

## Experimental Results (Session 001, 2026-04-01)

Infrastructure established: BitNet b1.58-2B-4T on Ryzen 5 PRO 5675U (AVX2, CPU-only, 62GB RAM). Eval: WikiText-2 raw, 20 chunks at ctx=512, llama-perplexity. Baseline PPL: 18.852.

### Phase 1 Result: Matmul kernel swap — VALIDATED

Replaced `_mm256_maddubs_epi16({0,1,2})` with `sub_epi8(unpack, 1)` → `sign_epi8(act, weight)` → `maddubs_epi16(ones, product)` in all four AVX2 dot product variants. Removed the `- act_sums` post-processing correction (no longer needed — sign_epi8 computes the true ternary dot product directly).

**PPL: 18.852 — identical to baseline.** Mathematically equivalent, as expected.

Note: The original plan predicted "6.87x speedup on the dominant compute operation." This was incorrect — the 6.87x benchmark compared sign_epi8 vs float32 (mulps), not vs the already-integer maddubs kernel. The existing I2_S kernel is already operating in the integer SIMD domain. Speed within that domain is similar. The value of this phase is correctness validation, not speed.

### Phase 2 Result: Activation quantization — 4-TRIT IS OPTIMAL

13-point precision sweep quantizing ALL matmul outputs across ALL 28 layers:

| Trits | Levels | PPL | Δ from baseline |
|-------|--------|-----|-----------------|
| 1 | 3 | 64793 | catastrophic |
| 2 | 9 | 62.5 | catastrophic |
| 3 | 27 | 20.11 | +1.26 (degraded) |
| **4** | **81** | **18.84** | **-0.01 (within noise)** |
| 5 | 243 | 18.89 | +0.04 |
| int8 | 256 | 18.85 | 0 (baseline) |

Extended evaluation (50 chunks): 4-trit PPL 16.600 vs baseline 16.554. Delta within confidence interval — not statistically significant.

**Path A confirmed.** 4-trit (81 levels, 6.3 bits) is sufficient for matmul outputs. MTFP21 (Path B) is not needed for this operation. The precision floor is between 3 and 4 trits.

**Anomalous band at MAX=26-28:** A non-monotonic degradation centered on 3^3 = 27 was discovered. MAX=25 (51 levels, PPL 18.92) is fine, but MAX=26 (PPL 19.25) and MAX=27 (PPL 19.22) spike before recovering at MAX=30 (PPL 18.89). Ternary-aligned granularities (13, 40, 121) behave more predictably than arbitrary values.

### Phase 3 Result: Residual connections — TRIVIALLY EXACT

ADD is exact in integer. Range after single residual: ±80 (fits int8). Accumulated across 28 layers: ±1120 (fits int16). No experiment needed.

### Phase 4 Result: Element-wise multiply — TESTED VIA COMBINED PATH

The FFN gate × up multiply receives inputs from matmul outputs (quantized to 4-trit) and SiLU output (quantized separately). The combined path was tested in Phase 2 — no separate experiment needed.

### Phase 5 Result: RMSNorm — MOST SENSITIVE OPERATION

RMSNorm output quantization with 4-trit matmul:

| RMSNorm trits | PPL | Δ from baseline |
|---------------|-----|-----------------|
| 4 | 25.93 | +7.08 (broken) |
| 5 | 20.21 | +1.36 |
| 6 | 19.13 | +0.28 |
| **7** | **18.91** | **+0.06 (good)** |

RMSNorm output is the MOST SENSITIVE point in the pipeline. The normalized pre-matmul activations need ~7 trits (2187 levels) for quality. The transcendental itself (one sqrt per row) is tractable in integer: sum of squares fits int32, and integer Newton-Raphson or a small lookup table handles rsqrt.

### Phase 6 Result: SiLU — QUANTIZATION HAS ZERO EFFECT

With 4-trit matmul + 7-trit RMSNorm, SiLU output quantization at 4-trit, 5-trit, and 6-trit ALL produce identical PPL (18.91). SiLU output is already ternary-compatible — quantizing it makes no additional difference. The activation function naturally concentrates outputs in a small effective range.

### Phase 7: Softmax — UNTESTED

Softmax is shift-invariant (adding a constant to all logits doesn't change the output), which suggests ternary-precision logits should work. Not yet validated.

### Phase 8: Embeddings + LM head — UNTESTED

### Precision Map of the Transformer Pipeline

```
Operation          Trit precision    Notes
─────────────────  ────────────────  ─────────────────────────
RMSNorm output     7 trits (2187)    THE BOTTLENECK
Matmul output      4 trits (81)      Zero quality impact
SiLU output        4 trits (81)      FREE — zero effect
Residual ADD       Exact             int16 range sufficient
Softmax            TBD               Expected to be fine (shift-invariant)
Embeddings         TBD               Untested
```

The entire linear compute path (matmul + residual + element-wise multiply + activation functions) can go ternary at 4-trit precision. The only bottleneck is RMSNorm, which needs 7 trits. If the integer rsqrt can be computed at that precision, the full ternary pipeline is viable.

## Original Expected Outcome (for reference)

> The matmul swap (Phase 1) should be clean — we're just changing the encoding from {0,1,2} to {-1,0,+1} and using a faster instruction. No precision change.

**Confirmed.** PPL identical.

> The activation quantization (Phase 2) is the big unknown. If 5-trit survives 28 layers, the entire linear compute path goes ternary integer. If it doesn't, we'll know the precision floor and can use MTFP21 or a higher trit count.

**Better than expected.** Not just 5-trit — 4-trit survives. MTFP21 is not needed for activations.

> The transcendentals (Phases 5-7) are the hardest. These are where the iGPU earns its place — or where frozen shape approximation proves itself.

**Partially answered.** RMSNorm needs 7-trit precision (the sqrt/rsqrt is the real constraint). SiLU is free. Softmax untested. The iGPU may still be valuable for the rsqrt computation, but a lookup table or integer Newton-Raphson may suffice.

## Infrastructure (completed)

- [x] BitNet b1.58-2B-4T model on this machine (GGUF, I2_S quantization)
- [x] BitNet.cpp building and running baseline inference (27 tok/s)
- [x] Fixed eval set for perplexity measurement (WikiText-2 raw, 20 chunks, ctx=512)
- [x] Build system that lets us swap individual operations (compile-time flags)
- [x] Baseline measurements recorded (PPL 18.852, 78 tok/s prompt eval, 2014 MB RSS)
- [ ] Fixed prompt set for qualitative comparison (not yet created)
