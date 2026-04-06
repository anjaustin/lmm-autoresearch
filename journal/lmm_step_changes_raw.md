# RAW: Step-Changes for Shirley — What Gets Us to the Next Level?

We're at 4.57 tok/s. The baseline float32 path is 22 tok/s. We're at 21% of baseline speed with a fundamentally different number system. The question isn't "how do we optimize another 5%" — it's "what changes the game?"

What I know about where the time goes:
- ~30% ternary matmul (sign_epi16, already multi-threaded in FFN, single-threaded in attention)
- ~14% FFN trivials (now native MTFP21 — scalar mul per element)
- ~14% FFN sub_norm (MTFP21 RMSNorm — chunked sum-of-squares + vectorized scale, but scalar rsqrt + scalar normalization loops)
- ~12% ggml boundary conversions (mtfp21_from_float / mtfp21_to_float)
- ~8% attention Q@K^T (chunked MTFP21 dot, scales with sequence length)
- ~7% each: attention norm, RoPE, residual

What scares me:
- The 4.5× gap to baseline might be fundamental — MTFP21 arithmetic is inherently more expensive per element than float32, and no amount of optimization closes that
- Or the gap might be almost entirely from things we're doing wrong, not things MTFP21 does poorly

What's probably wrong with my current thinking:
- I've been optimizing within the current architecture (SIMD lanes, threading, conversion removal) instead of questioning the architecture itself
- The matmul is sign_epi16 at 16 lanes. The baseline uses sign_epi8 at 32 lanes. We're at HALF the matmul throughput by design. Is that necessary?
- The MTFP21 scalar ops (mul, add) are ~50ns each. Float32 mul is ~0.5ns (FMA). That's 100× per operation. Even though we do fewer operations in some paths, the per-op cost dominates.

What would a step-change look like?
- Going from 4.57 to 10+ tok/s (2× or more improvement)
- Not incremental tuning — a structural change that removes an entire category of overhead

Ideas I haven't explored:
1. The ggml boundary — what if we eliminated it entirely? shirley_forward() that owns the data path. No mtfp21_from_float / mtfp21_to_float per layer (saves ~12%).
2. The matmul at int8 instead of int16 — we proved int8 loses PPL when quantizing BETWEEN ops. But at the matmul wire (post-normalization, tight range), int8 might be sufficient. This would double the matmul throughput (32 lanes vs 16).
3. Multi-threading the attention — the split-node approach was right in principle but had an implementation bug. If fixed, it would multi-thread the 30% that's currently single-threaded.
4. The MTFP21 per-element ops — can they be vectorized? The mantissa is int32 (8 AVX2 lanes). The multiply is int32×int32→int64 (4 lanes via mul_epi32). The normalization is the serial bottleneck (trit-shift loop).
5. Eliminating normalization entirely — what if we tracked mantissa range per vector (block style) and only normalized at specific checkpoints?
6. Rethinking what "between matmuls" means — the FFN trivials (relu, square, multiply) could be fused into the matmul output processing. Instead of matmul → normalize → relu → normalize → square → normalize → multiply → normalize, do it all in one pass with deferred normalization.
7. The iGPU — the original Shirley thesis routes EXP to the iGPU. Softmax has 156M EXP operations per forward pass. At 4 cycles per V_EXP_F32 on RDNA CUs vs ~200ns per mtfp21_exp on CPU...

Three open questions:
1. Is the 100× per-op cost of MTFP21 vs float32 the fundamental bottleneck, or is it the architecture around the ops?
2. Can the matmul go back to int8 (sign_epi8, 32 lanes) without quality loss, now that we keep MTFP21 between matmuls?
3. What does the pipeline look like if we remove ggml entirely?
