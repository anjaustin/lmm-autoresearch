# REFLECT: BitNet Session 001

## The Core Insight

**The transformer's information content has a sharp ternary floor — and we found it.**

This session wasn't about optimization. It was about measurement. We held a ruler up to the internal representations of a 2-billion-parameter language model and read off the minimum precision at which the representations still function. The answer is 4 balanced ternary trits for most operations, 7 at normalization boundaries. Below 4 trits, the model undergoes a phase transition from functional to catastrophic within one trit of precision. Above 4 trits, additional precision adds nothing measurable.

That's not an engineering finding. It's a characterization of the model's information geometry. The question "how many bits do you need?" has a specific, measurable answer — and it's aligned with balanced ternary arithmetic.

## Resolving the Tensions

### Tension 1: Speed vs. correctness framing
The plan was motivated by speed: sign_epi8 is faster than float. The finding is that the existing kernel is already integer — sign_epi8 isn't faster, it's more semantically honest. 

**Resolution:** Reframe the project from "make BitNet faster with ternary" to "characterize the ternary structure that already exists in BitNet." The speed benefit comes not from replacing one integer instruction with another, but from the precision finding: if 4-trit suffices, you need 4x less memory bandwidth for activations, which IS a speed benefit — just not the one we predicted.

### Tension 2: Uniform vs. mixed precision
4-trit for matmul and SiLU, 7-trit for RMSNorm. Mixed precision complicates the pipeline.

**Resolution:** This isn't a complication — it's the natural structure. The six primes (ADD, MUL, EXP, LOG, MAX, CONST) route to different substrates. Precision requirements route to different widths. Mixed precision isn't a compromise; it's what emerges when you let each operation declare its own information requirement. The MTFP21 representation (16-trit mantissa + 5-trit exponent) was designed for this — it can represent both 4-trit and 7-trit values in the same format, just with different numbers of significant trits.

### Tension 3: Sweet spot vs. cliff edge
4-trit works. 3-trit breaks. One trit of margin.

**Resolution:** The sharp floor is actually reassuring, not alarming. It means the model's information content is well-defined — there's a clear boundary between signal and noise. Gradual degradation would mean the precision requirement is fuzzy and context-dependent. A sharp floor means: this is how much information the model actually uses, period. The margin question becomes: do different models have different floors? If the floor scales with model quality/size, we need to characterize it per model. If it's universal for ternary-weight transformers, 4 trits is the answer.

### Tension 4: Simulation vs. reality
Every finding uses quantize-dequantize in float32, not actual integer arithmetic.

**Resolution:** The simulation is measuring the right thing — precision loss — but not the only thing that matters. Integer accumulation could expose overflow issues (mitigated by the int16/int32 widening in the existing kernel). Rounding behavior differs between float roundf() and integer truncation. These are real concerns, but they're bounded: the simulation shows 4-trit precision is sufficient, and the actual integer path would have AT LEAST the same precision (because integer arithmetic is exact within its range, while the float round-trip introduces its own rounding). The simulation is a lower bound on quality, not an approximation. Real integer compute should match or exceed the simulation.

Actually — wait. That's not quite right. The simulation quantizes globally per block (find max, scale, round). Real integer compute wouldn't have a per-block scale factor built in — the values would be actual ternary integers. The interaction between the quantization scale and the subsequent operation matters. If the next operation is a matmul with ternary weights, the scale factor would need to be tracked and applied post-multiply. This is basically the same structure as the existing activation quantization (which already has act_scale). So the plumbing exists — it's just a question of whether 4-trit quantization with the existing scale infrastructure produces the same results as the simulation.

This is testable without building a full integer pipeline. We could quantize activations to 4-trit integers, pass those integers through the existing matmul kernel (which already handles int8 × ternary), and check whether the results match. I flag this as the most important validation experiment for the next session.

### Tension 5: Characterization vs. engineering
The research question is answered. The engineering question reduces to integer rsqrt.

**Resolution:** These aren't sequential — they're parallel. The characterization result (4-trit is sufficient) enables engineering decisions immediately. The rsqrt problem is well-studied in integer signal processing. A 7-trit lookup table has 2187 entries — small enough to fit in L1 cache. Newton-Raphson in 7-trit fixed-point arithmetic converges in 2-3 iterations from a table-seeded initial guess. This is not a research problem; it's an afternoon of implementation.

## Hidden Assumptions Challenged

### "More precision is always better"
The non-monotonic anomaly at MAX=27 directly contradicts this. 55 levels (MAX=27) is worse than 41 levels (MAX=20). Precision is not a monotone function of level count. The alignment between quantization granularity and the model's internal structure matters. This has implications for all quantization research, not just ternary.

### "Activation functions are the hard part of going ternary"
SiLU turned out to be free. Zero effect from quantization. The assumption that transcendental operations would be the engineering bottleneck was wrong for SiLU, partially right for RMSNorm (the rsqrt, not the per-element operations), and untested for softmax. The difficulty map was inverted from our prediction.

### "The model uses all 32 bits of float precision"
It uses about 6.3 bits per activation element. The other 25.7 bits are noise. This isn't specific to ternary models — it's a measurement of the model's actual information density. Float32 is not an information-theoretic requirement; it's a hardware convenience. The model's representations are far more compressible than the storage format suggests.

### "Denoising through quantization is a small-scale phenomenon"
The MNIST result was on 784-dimensional feature vectors with a 10-class classification task. Extrapolating to 2560-dimensional hidden states in a 28-layer autoregressive language model was a leap. It held. The denoising effect is not a property of the task scale — it's a property of the relationship between ternary quantization and the statistics of neural network activations. This suggests it would hold for larger models too, though that needs validation.

## What Would This Look Like If It Were Easy?

If the full ternary inference pipeline were trivial to build, you'd:
1. Load the model's ternary weights as int8 {-1,0,+1}
2. Quantize input activations to 4-trit at each layer
3. Matmul via sign_epi8 + accumulate (already working)
4. RMSNorm with integer rsqrt at 7-trit precision (one lookup table)
5. SiLU: just quantize the output to 4-trit (free)
6. Softmax: compute in float, quantize output (or find it's free like SiLU)
7. Output logits: compute in float for the final softmax/sampling

Step 4 is the only non-trivial step. Everything else either works already or is a quantize-dequantize call. The "hard" problem is one lookup table and a Newton-Raphson iteration.

## The Structure Beneath the Content

The individual findings (4-trit works, SiLU is free, RMSNorm is the bottleneck) are symptoms. The underlying structure is:

**Ternary-weight models produce ternary-compatible activations.**

The weights constrain the information content of the activations. When every weight is {-1, 0, +1}, the output of a matmul is a sum/difference of input elements. The information capacity of that sum is bounded by the information in the weight pattern, not by the precision of the arithmetic. Float32 activations are like recording a ternary signal on vinyl — the medium has more resolution than the source material contains.

This is why 4 trits work. This is why SiLU is free (it's a continuous function applied to an already-low-information signal — the output inherits the input's information bound). This is why RMSNorm needs more precision (normalization amplifies relative differences, temporarily increasing the effective information content before the next matmul collapses it back down).

The precision map of the pipeline isn't a list of independent findings. It's a measurement of information flow through the transformer: low at matmul outputs (4 trits), briefly high at normalization outputs (7 trits), low again after the next matmul (4 trits). The information pulses at normalization boundaries and rests everywhere else.
