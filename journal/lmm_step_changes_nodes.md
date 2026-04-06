# NODES: Step-Changes for Shirley

## Node 1: The 100× per-op gap is misleading
Float32 FMA: ~0.5 ns per multiply-add. MTFP21 mul: ~50 ns. That's 100×. But the MODEL does different amounts of work per format. Float32 does: RMSNorm (vectorized), quantize to int8 (per-element), matmul (int8×ternary), dequantize to float32, scale multiply. MTFP21 does: RMSNorm (vectorized), block-align (per-element trit-shift), matmul (int16×ternary), normalize (per-element). The quantize and block-align steps are comparable. The dequantize and normalize are comparable. The per-op cost difference matters for BETWEEN-matmul ops (trivials), not for the matmul-adjacent ops.

## Node 2: The matmul throughput is half
sign_epi8: 32 lanes. sign_epi16: 16 lanes. Same instruction latency. Half the throughput. The matmul is 30% of time. Doubling its throughput would give ~15% speedup — from 4.57 to ~5.3 tok/s. Not a step-change by itself, but stacks with other improvements.

## Node 3: The ggml boundary is pure waste
12% of time converting MTFP21 ↔ float at the custom op boundary. This exists because ggml passes float tensors. If we owned the data path (shirley_forward()), this disappears entirely. 12% recovery → 4.57 × 1.12 = ~5.1 tok/s.

## Node 4: The attention is single-threaded
The attention matmuls (QKV + wo = ~30% of attention time) and the attention body (Q@K^T + attn@V = ~9% of attention time) are all on thread 0. Multi-threading just the QKV+wo matmuls across 6 threads would save ~25% of attention time. Attention is ~45% of total. So: 45% × 25% = ~11% overall. ~5.1 tok/s.

## Node 5: The normalization loops are serial
Every MTFP21 multiply produces a result that may need trit-shifting (while |m| > MANT_MAX { /=3; exp++ }). This loop is 1-2 iterations on average, but it's SERIAL — can't be vectorized because each element needs a different number of shifts. For 414K mtfp21_mul calls per token (trivials alone), 2 iterations × ~10 ns = ~8 ms. This is the dominant cost in the native MTFP21 trivials.

## Node 6: The matmul at int8 might work now
The original int8 quantization lost PPL because we quantized BETWEEN every op — 90 quantization events per token across 30 layers. Now we keep MTFP21 between ops. The only quantization is at the matmul wire, where the data is post-normalization (tight range). The test showed 0 values crushed on post-normalization data at int16. Would int8 also crush 0 values? The range is ±80 with 5-trit quantization vs ±29524 with MTFP16. The post-normalization data has already been shown to fit int8 without PPL loss (the original Option D validation: PPL 18.888 at ±80, baseline 18.852).

## Node 7: Fusion eliminates normalization
matmul outputs int32. Currently: normalize to MTFP21 → relu (check sign) → square (mtfp21_mul) → multiply (mtfp21_mul). Each step normalizes. If we fused: int32 gate output → check sign (relu on int32 — just check sign bit) → square (int32×int32→int64) → multiply by up (int64×int32→int64) → normalize ONCE to MTFP21. One normalization instead of three. Saves 2/3 of the trit-shift loops in trivials.

## Node 8: The iGPU is untapped
The machine has an integrated GPU with hardware EXP/LOG. Softmax uses 156M EXP ops per forward pass. Currently on CPU via mtfp21_exp (LUT + interp, ~200 ns each). On iGPU: V_EXP_F32 at ~4 cycles on massively parallel CUs. Shared memory (APU) means no PCIe transfer. This is the original Shirley thesis — route EXP to iGPU.

## Node 9: Remove ggml entirely
shirley_forward() replaces build_bitnet_158(). No graph. No tensor allocation. No custom op overhead. The data path is: embedding → 30 × (attention + FFN) → output. A simple loop. The MTFP21 state persists between layers without conversion. The only float is at the model boundaries (embedding in, logits out).

## Tensions:
- Node 2 vs Node 6: going back to int8 matmul doubles throughput BUT we proved the matmul precision matters. The question is whether post-normalization data is tight enough for int8.
- Node 5 vs Node 7: normalization loops are the bottleneck, fusion eliminates them. But fusion means operating on raw int32/int64 instead of MTFP21 — which is moving AWAY from native MTFP.
- Node 3 vs Node 9: removing ggml boundary (12%) is easier than removing ggml entirely. But removing ggml entirely gives more (no tensor alloc, no graph dispatch, no custom op overhead).
- Node 7: fusing the trivials is operating on raw integers — is this betraying the vision of native MTFP? Or is it the correct implementation of MTFP where normalization is deferred, not eliminated?
