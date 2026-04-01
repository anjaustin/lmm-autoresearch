# RAW: BitNet Session 001 — Ternary Conversion Findings

## Stream of Consciousness

We set out to answer a straightforward question: can you replace float32 operations with ternary integer operations in a real language model, one at a time, without breaking it? We had a plan — eight phases, surgical swaps, measure after each one. What we got back was stranger and more interesting than the plan anticipated.

The first thing that happened is that the plan's headline claim was wrong. The 6.87x speedup from sign_epi8 was measured against float32 mulps — not against the already-integer maddubs kernel that BitNet actually uses. The existing kernel is already in the integer SIMD domain. We're not replacing float with integer. We're replacing one integer encoding ({0,1,2} + maddubs) with another ({-1,0,+1} + sign_epi8). The speed story evaporated on contact with the actual codebase. But the correctness story held — sign_epi8 produces mathematically identical results, and more importantly, it computes the TRUE ternary dot product directly, without the offset correction that the {0,1,2} encoding requires. The existing code computes w*a where w is shifted by +1 from the real ternary value, then subtracts the activation sum in post-processing to compensate. sign_epi8 just... does the right thing. No correction needed.

That's interesting but not the finding that matters. The finding that matters is what happened when we started quantizing activations.

The plan said Phase 2 was "the big unknown." Would 5-trit quantization survive 28 layers of compounding? The MNIST result from Shirley Phase 1 showed denoising at the feature level — quantization to 5 trits removed noise rather than signal. But MNIST features are not LLM activations. A 784-dimensional image vector is not a 2560-dimensional hidden state flowing through attention and feedforward networks. The extrapolation was a hope, not a prediction.

It survived. Not just 5-trit. Four-trit. 81 levels. 6.3 bits of precision on every matmul output across every layer of a 28-layer, 2-billion-parameter transformer. The perplexity didn't degrade. On the 20-chunk eval it actually improved slightly (18.843 vs 18.852), though the 50-chunk extended eval showed the difference is within the confidence interval — not statistically significant. The honest statement is: 4-trit quantization has no measurable effect on quality. Not "it improves quality." Not "it barely degrades." No measurable effect.

But then the sweep got weird. The response is non-monotonic. You'd expect: more levels = better quality, fewer levels = worse quality, smooth curve. That's not what happened. 81 levels (4-trit aligned): fine. 55 levels (MAX=27): spike to PPL 19.22, a 0.36 degradation that's clearly outside noise. 51 levels (MAX=25): fine again at 18.92. The spike is centered exactly on 3^3 = 27. And it's not a point — it's a band. MAX=26 spikes too (19.25). MAX=28 is elevated (19.02) but recovering. By MAX=30 we're back to normal.

I don't fully understand why. The weights are ternary. The activations emerge from ternary weights acting on quantized inputs. There might be structure in the activation distributions that aligns with powers of 3 — and quantizing at exactly that boundary creates destructive interference. Or it might be simpler: certain quantization step sizes happen to round critical values across a threshold that matters for attention or routing. Either way, the ternary-aligned granularities (13, 40, 121) all produce cleaner results than nearby non-aligned values. That's a pattern worth understanding.

Below the floor — 3 trits (27 levels), 2 trits (9 levels), 1 trit (3 levels) — degradation is steep and then catastrophic. 3-trit: PPL 20.1, degraded but the model still forms sentences. 2-trit: PPL 62.5, the model is babbling. 1-trit: PPL 64,794, the model is dead. The cliff between "works" and "doesn't work" is sharp. It's not a graceful degradation — it's a phase transition somewhere between 27 and 41 levels.

Then the RMSNorm result. Everything up to this point was about matmul outputs — the values AFTER the weight-activation multiply. RMSNorm output is the value BEFORE the next multiply — the normalized activation that feeds directly into the weight matrix. And it turns out this is a completely different sensitivity. 4-trit on RMSNorm output: PPL 25.93, broken. 5-trit: 20.21, degraded. 6-trit: 19.13, marginal. 7-trit: 18.91, good. The pre-matmul activation needs roughly 2x more trit precision than the post-matmul activation. That makes intuitive sense — normalization concentrates values near zero, and the relative precision at small magnitudes matters more. But I didn't predict the magnitude of the gap. I expected maybe one extra trit. It's three.

And then SiLU. We tested 4-trit, 5-trit, and 6-trit quantization on the SiLU activation function output. All three produced identical PPL: 18.9129. Not "similar." Not "within noise." Identical to six decimal places. The SiLU output simply does not care about quantization precision down to 81 levels. The activation function concentrates its output in a way that's already representable in very few levels. This was the most surprising result of the session. The operation we expected to be one of the harder transcendentals turned out to be free.

So now we have a precision map:
- Matmul outputs: 4 trits
- SiLU outputs: 4 trits (free — zero effect)
- Residual ADD: exact in integer
- RMSNorm outputs: 7 trits (the bottleneck)
- Softmax: untested
- Embeddings: untested

The entire linear compute path is 4-trit. The only thing that needs more precision is the one operation that involves a square root. And even that — the transcendental part of RMSNorm is one rsqrt per row, computed once per 2560 elements. The per-element work (squaring, accumulation, scaling) is all integer-friendly. It's just the final rsqrt that needs to land within 7-trit precision of the float32 result.

What I keep thinking about: the model was trained with ternary weights but float32 activations. We just showed that the activations don't need to be float32. They can be 6.3 bits. The model's information content — the part that matters for predicting the next token — fits in 4 balanced ternary trits per activation element. The other 17+ bits of float32 precision are carrying noise, not signal.

That's the Shirley thesis in one measurement. Computation has a natural ternary form. The hardware has been speaking ternary via sign_epi8 for 15 years. And now we've shown that the activations — the learned, emergent, high-dimensional representations inside a transformer — are ternary-compatible too. Not by design. Not by training. By nature.

## Open Questions

1. Why does the non-monotonic spike happen at 3^3? Is there actual ternary structure in the activation distributions, or is this a coincidence of quantization step sizes?
2. The RMSNorm sensitivity gap (7 vs 4 trits) — is this specific to RMSNorm, or would any normalization operation show the same? What about LayerNorm?
3. Softmax is untested. It's shift-invariant, so ternary logits should work. But "should" isn't "does."
4. The extended eval (50 chunks) showed the 4-trit "improvement" is within noise. With even more data, does a real degradation emerge? Or is it genuinely zero?
5. We tested simulated quantization (quantize-then-dequantize in float). Real ternary compute would have different numerical properties — rounding, accumulation, overflow. Does the simulation faithfully predict the real behavior?
6. What about other models? Is 4-trit universal for ternary-weight transformers, or specific to this architecture/size?
7. The SiLU result — why is it free? Is this specific to SiLU, or would GeLU/ReLU/etc. show the same insensitivity?
8. Can the integer rsqrt for RMSNorm be done at 7-trit precision using Newton-Raphson or a lookup table? What's the engineering cost?

## First Instincts

- The non-monotonic spike is the most intellectually interesting finding. It suggests structure we don't understand yet.
- The SiLU result is the most practically important finding. It eliminates an entire category of engineering work.
- The RMSNorm bottleneck is the clearest next engineering target. Solve rsqrt at 7-trit precision and the full pipeline goes ternary.
- The 4-trit result should be validated on at least one other model before making general claims.
- The simulation (quantize-dequantize) needs to be compared against actual integer compute before we trust it as a proxy.
