# NODES: BitNet Session 001

## Extracted Nodes

### Node 1: The speed claim was wrong, the correctness claim was right
The plan predicted 6.87x speedup from sign_epi8. That benchmark was against float32, not against the existing integer kernel. The I2_S kernel already operates in AVX2 integer SIMD. sign_epi8 is not faster than maddubs in this context — it's mathematically cleaner. It computes the true ternary dot product without the offset correction. The value of Phase 1 is validation, not acceleration.

**Tension with initial plan:** The plan was motivated by speed. The finding is about semantic correctness. This reframes the entire project — we're not making BitNet faster, we're making it *more ternary*.

### Node 2: 4-trit is the activation precision sweet spot
81 levels (6.3 bits) on all matmul outputs across 28 layers produces no measurable quality degradation. This is 4x less precision than the current int8 activation quantization (256 levels). The model's information content per activation element fits in 4 balanced ternary trits.

**Connection to Shirley thesis:** The MNIST finding (5-trit lossy beats 6-trit lossless) extrapolated to a 2B-param LLM. The denoising hypothesis holds at scale.

### Node 3: The non-monotonic anomaly at 3^3
Quantization at MAX=26-28 (centered on 3^3 = 27) produces an anomalous PPL spike. Nearby values (MAX=25, MAX=30) are fine. Ternary-aligned granularities (13, 40, 121) produce cleaner results than arbitrary values. This is either:
- (a) Structural: activation distributions have ternary-aligned features due to ternary weights
- (b) Numerical: specific step sizes create aliasing at critical thresholds
- (c) Coincidence

**Tension with Node 2:** Node 2 says "4-trit works." Node 3 says "but the choice of quantization level matters in non-obvious ways." The sweet spot isn't just about precision — it's about alignment with the model's internal structure.

### Node 4: Pre-matmul activations are 3x more sensitive than post-matmul
RMSNorm output (pre-matmul) needs 7 trits. Matmul output (post-matmul) needs 4 trits. The normalized, pre-multiply activation carries more information per element. Normalization concentrates values near zero where relative precision matters most.

**Dependency:** This means the ternary pipeline is NOT uniform precision. It needs mixed precision — 4 trits for most operations, 7 trits at the normalization boundary. This complicates the "everything is 4-trit" narrative.

### Node 5: SiLU is quantization-invariant
Quantizing SiLU output to 4, 5, or 6 trits produces identical PPL (18.9129). The activation function naturally produces outputs that are already representable in very few levels. This eliminates SiLU as an engineering concern for ternary conversion.

**Tension with plan:** The plan listed SiLU as Phase 6 (hard, transcendental). It turned out to be the easiest finding of the session — the operation is already ternary-compatible without any modification.

### Node 6: The precision floor is a cliff, not a slope
3-trit (27 levels): degraded but functional (PPL 20.1). 2-trit (9 levels): catastrophic (PPL 62.5). 1-trit (3 levels): dead (PPL 64,794). The transition from "works" to "broken" spans less than one trit of precision. This is a phase transition, not a graceful degradation.

**Connection to Node 2:** The sweet spot at 4 trits is only one trit above the cliff edge. There's no large margin of safety. If some operation or model variant needs slightly more precision, the fallback is immediate.

### Node 7: Simulated quantization may not predict real compute behavior
All experiments used quantize-dequantize in float32 — a round-trip that simulates precision loss but doesn't exercise actual integer arithmetic. Real ternary compute would accumulate rounding errors differently, have different overflow characteristics, and might expose issues the simulation doesn't capture.

**Dependency on all other nodes:** Every precision finding (Nodes 2, 4, 5, 6) was measured via simulation. If the simulation is unfaithful, the precision map needs re-measurement with real integer paths.

### Node 8: RMSNorm is the single engineering bottleneck
The entire linear compute path (matmul, residual, activation functions) works at 4-trit. Only RMSNorm needs higher precision (7 trits). The transcendental part (rsqrt) is computed once per row — not per element. Integer Newton-Raphson or a lookup table may suffice. This is a well-defined, bounded engineering problem.

**Connection to Shirley six primes:** RMSNorm decomposes as `x * EXP(CONST(-0.5) * LOG(MUL(x, x).mean()))`. The LOG and EXP are the primes that route to iGPU. This is exactly the substrate routing the Shirley thesis predicted — CPU handles ADD/MUL/MAX, iGPU handles the transcendental.

### Node 9: The plan's phase ordering was wrong
The plan ordered phases by "easiest and most impactful first" — matmul, then activations, then residuals, then element-wise multiply, then RMSNorm, SiLU, softmax, embeddings. In practice:
- Phase 4 (element-wise multiply) was tested implicitly via Phase 2 — no separate experiment needed
- Phase 6 (SiLU) was trivially free — should have been tested earlier, not later
- Phase 5 (RMSNorm) was the critical finding — should have been tested immediately after Phase 2, not after Phase 4

The actual information-optimal ordering would have been: matmul → activation quant → RMSNorm → SiLU → done. Four experiments instead of eight phases.

### Node 10: The model was already ternary — we just measured where
The weights are ternary by design. The activations don't NEED to be float32 — 4 trits suffices. SiLU is already ternary-compatible. Residuals are exact in integer. The model was thinking in ternary all along. Float32 was overhead, not information. The 17+ bits of float32 precision above the 6.3-bit ternary floor were carrying noise.

This isn't an optimization finding. It's a characterization finding. We didn't make the model ternary. We measured that it already was.

## Key Tensions

1. **Speed vs. correctness framing** (Node 1 vs. plan): The project was motivated by speed. The findings are about precision and structure. The value proposition needs reframing.

2. **Uniform vs. mixed precision** (Node 2 vs. Node 4): "Everything is 4-trit" is the clean narrative. Reality is mixed: 4 trits for most operations, 7 trits at normalization. The pipeline needs precision transitions.

3. **Sweet spot vs. cliff edge** (Node 2 vs. Node 6): 4-trit is optimal. 3-trit is broken. One trit of margin. This is either alarming (fragile) or informative (the model's information content has a sharp lower bound).

4. **Simulation vs. reality** (Node 7 vs. everything): Every finding rests on simulated quantization. Actual integer compute is the untested foundation.

5. **Characterization vs. engineering** (Node 10 vs. Node 8): The research question ("is ternary viable?") is answered. The engineering question ("can we build it?") reduces to one operation: integer rsqrt at 7-trit precision.
