# SYNTHESIS: Step-Changes for Shirley

## The Path to 2× Speed (4.57 → ~9 tok/s)

Five structural changes, each removing an entire category of overhead. They stack.

### 1. Fused Trivials with Deferred Normalization

**What:** The matmul outputs raw int32 + block exponent. ReLU checks the sign bit (no normalization). Square is int32×int32→int64 (no normalization). Multiply is int64×int32→int64 (no normalization). ONE normalization to MTFP21 at the end, before sub_norm needs it.

**Why it works:** The intermediate int64 values are MTFP with extended mantissa. The exponent tracks correctly (double for square, add for multiply). The normalization at the end is identical to what we do now — just once instead of three times.

**Impact:** Eliminates 2/3 of the trit-shift loops in the trivials path. Estimated ~10% overall.

**Files:** `shirley_mtfp16_matmul.h` (output raw int32 + block_exp option), `shirley_ffn.cpp` (fused trivials section).

### 2. Remove ggml Boundary

**What:** `shirley_forward()` replaces the ggml graph for the layer loop. The MTFP21 state persists between attention and FFN custom ops without converting to/from float. The embedding lookup and LM head stay as ggml calls at the model boundary.

**Why it works:** The 60 conversion points per token (2 per layer × 30 layers) each touch 2560 elements of mtfp21_from_float / mtfp21_to_float. These are pure waste — the data is MTFP21 on both sides.

**Impact:** ~12% overall.

**Files:** New `shirley_forward.cpp`, modified `llama.cpp` decode path.

### 3. Attention Multi-Threading (Split-Node, Fixed)

**What:** Debug the split-node regression from `85a616e`. The approach (5 graph nodes per layer, ggml manages barriers) is correct in principle. The bug is likely in initialization or data lifetime, not synchronization.

**Why it works:** The QKV + wo matmuls are row-parallel. Multi-threading across 6 cores saves ~25% of attention time.

**Impact:** ~11% overall.

**Files:** `shirley_attn.cpp` (split-node callbacks), `llama.cpp` (graph wiring).

### 4. int8 Matmul (32 Lanes)

**What:** Test whether the matmul can use sign_epi8 (32 lanes) instead of sign_epi16 (16 lanes) at the matmul wire, now that MTFP21 is preserved between matmuls. The post-normalization data was validated at int8 range ±80 (Option D: PPL 18.888 vs baseline 18.852).

**Why it might work:** The original int8 quality loss was from quantizing BETWEEN every op (90 quantization events). Now we quantize ONLY at the matmul wire (2 per layer). The data is post-RMSNorm (tight range) in both cases.

**Why it might not:** The MTFP21→int8 block alignment compresses 16-trit mantissa to 7-trit int8. The MTFP21→int16 block alignment compresses to 10-trit int16. The extra 3 trits in int16 might matter for the matmul accumulation precision.

**Impact:** If it works, doubles matmul throughput. ~15% overall.

**Validation needed:** PPL test + generation quality on 3+ prompts.

**Files:** New `shirley_mtfp8_matmul.h` or modify `shirley_mtfp16_matmul.h` to support int8.

### 5. iGPU EXP Dispatch

**What:** Route the softmax EXP operations to the integrated GPU's hardware transcendental unit (V_EXP_F32). Shared memory on the APU eliminates PCIe transfer.

**Why it works:** CPU mtfp21_exp: ~200 ns per call (LUT + interp). iGPU V_EXP_F32: ~4 cycles per op on massively parallel CUs. For 156M EXP ops per forward pass, this is significant at longer sequences.

**Impact:** ~5% at current short sequences, scales with sequence length.

**Files:** New iGPU dispatch layer. Significant engineering.

## Priority Order

1. **Fused trivials** — lowest risk, highest confidence, builds on the remediation we just did
2. **Remove ggml boundary** — medium engineering, high payoff, clean architecture
3. **Attention multi-threading** — debug the split-node regression (known approach, unknown bug)
4. **int8 matmul** — needs PPL validation before committing
5. **iGPU EXP** — largest engineering effort, lower short-sequence impact

## Combined Estimate

Conservative (items 1-3 only): 4.57 × 1.10 × 1.12 × 1.11 ≈ **6.25 tok/s**
Aggressive (all 5): 4.57 × 1.10 × 1.12 × 1.11 × 1.15 × 1.05 ≈ **7.5 tok/s**

From 21% of baseline to 28-34% of baseline. Not parity, but a meaningful fraction — and the architecture is fundamentally different (ternary/MTFP21 vs float32).
