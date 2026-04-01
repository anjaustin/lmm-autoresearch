# DO THIS NEXT

## The One Thing That Matters

Wire `shirley_rmsnorm_ternary` as the bridge between the float32 domain and the int8 domain inside the BitNet inference pipeline. RMSNorm takes float32 from the residual connection, outputs int8 directly into the matmul's activation buffer, replacing the current `ggml_compute_forward_rms_norm_f32` + `quantize_row_i8_s` two-step with a single fused operation.

This is where the int8 data path starts for real. Everything built so far is either validated-in-pipeline (sign_epi8 matmul) or validated-standalone (kernels, MTFP21). The bridge connects them.

## What Exists Today

### In the pipeline (running in BitNet inference):
- **sign_epi8 matmul kernel** — all 7 ternary matmuls per layer use `_mm256_sign_epi8` instead of `_mm256_maddubs_epi16`. PPL 18.852, identical to baseline. File: `bitnet/src/ggml-bitnet-mad.cpp`
- **Bridge infrastructure** — `ggml_tensor` extended with `shirley_i8` (void*), `shirley_scale` (float), `shirley_bridge` (int32). wo and ffn_down matmuls tagged with `shirley_bridge=1`. File: `bitnet/3rdparty/llama.cpp/ggml/include/ggml.h` line 615-619, `bitnet/3rdparty/llama.cpp/src/llama.cpp` in `build_bitnet_158()`
- **Deferred materialization** — `shirley_bridge_materialize()` function converts float32 tensor data to int8 side-buffer on demand. Single-threaded, global per-tensor scale. File: `bitnet/3rdparty/llama.cpp/ggml/src/ggml.c` after the Shirley AVX2 RMSNorm inline function
- **Banner** — prints `shirley: pipeline active` to stderr on first RMSNorm call

### Standalone (validated but not in the pipeline):
- **`shirley_rmsnorm_ternary`** — int8→int8, zero float, 417 ns for n=2560, uses MTFP21 integer rsqrt (LUT+NR), Q15 fixed-point scale via `_mm256_mulhrs_epi16`. With gamma via Q14 scalar fallback. 31/31 tests pass. File: `bitnet/shirley_kernels.h`
- **`shirley_rmsnorm_f32`** — float32→float32 with gamma, AVX2, 430 ns. 8x faster than ggml scalar in standalone benchmark but NOT faster in the pipeline (ggml's `ggml_vec_scale_f32` is already SIMD-vectorized with loop unrolling). File: `bitnet/shirley_kernels.h`
- **`shirley_rmsnorm_quantize`** — float32→int8 fused (norm + gamma + quantize in one pass), 9.8 us. Returns quantization scale. File: `bitnet/shirley_kernels.h`
- **Ternary primitives** — `shirley_relu_i8` (30 ns), `shirley_square_i8_to_i16` (75 ns), `shirley_residual_add_i16` (136 ns), `shirley_requantize_i16_to_i8`, `shirley_mul_i16_i8_to_i8`. All tested, all correct. File: `bitnet/shirley_kernels.h`
- **Post-matmul rescale** — `shirley_rescale_i32_to_i8` and `shirley_rescale_i32_scaled_to_i8`. int32→int8 quantization. File: `bitnet/shirley_kernels.h`
- **MTFP21** — full arithmetic library (add, mul, div, rsqrt, conversion). 102/102 tests. Integer-only rsqrt via 256-entry LUT + 2 Newton-Raphson iterations. File: `bitnet/shirley_mtfp21.h`

### Known cost:
- 5% prompt eval overhead from the 16-byte `ggml_tensor` struct extension (cache alignment shift across all tensors)

## The Integration: Step by Step

### What the pipeline does today (per RMSNorm + matmul pair):

```
Step 1: ggml_compute_forward_rms_norm_f32()
        Input:  float32 tensor (from residual ADD or previous layer)
        Output: float32 tensor (normalized)
        Where:  ggml.c:12184, called from dispatch at ggml.c:18315
        
Step 2: ggml_mul(cur, gamma_weights)
        Input:  float32 (from step 1) × float32 (learned gamma)
        Output: float32 tensor (normalized + scaled)
        Where:  llama.cpp:9512 inside llm_build_norm()

Step 3: quantize_row_i8_s()
        Input:  float32 tensor (from step 2)
        Output: int8 buffer in params->wdata + act_scale + act_sum
        Where:  ggml.c:13280 inside ggml_compute_forward_mul_mat()
        Note:   This happens at the START of the next matmul, not as a separate op

Step 4: ggml_vec_dot_i2_i8_s() (the ternary matmul)
        Input:  int8 activations (from step 3) × 2-bit ternary weights
        Output: int32 dot products → float32 via (raw/act_scale)*weight_scale
        Where:  ggml.c:12595-12640, ggml-bitnet-mad.cpp
```

### What the pipeline should do (the bridge):

```
Step 1+2+3 FUSED: shirley_rmsnorm_quantize() or shirley_rmsnorm_ternary()
        Input:  float32 tensor (from residual ADD)
        Apply:  RMSNorm + gamma + quantize to int8 at range ±80
        Output: int8 buffer + quantization scale
        Where:  Replace ggml_compute_forward_rms_norm_f32 + ggml_mul + quantize_row_i8_s

Step 4: ggml_vec_dot_i2_i8_s() (unchanged)
        Input:  int8 activations (from fused step) × 2-bit ternary weights
        Output: int32 dot products → float32 via rescale
```

The key: steps 1, 2, and 3 currently produce three intermediate float32 tensors (normalized, gamma-scaled, quantized). The fused version produces one int8 buffer directly. Three float32 memory passes become one int8 pass = 12x less memory traffic for the activation data.

### Where to make the change

#### Option A: Replace at the RMSNorm dispatch level

Modify `ggml_compute_forward_rms_norm_f32()` in `ggml.c:12184` to detect when the RMSNorm output feeds a ternary matmul (via the graph structure or a flag) and produce int8 output instead of float32.

Problem: the RMSNorm doesn't know what consumes its output. The ggml compute functions don't have access to the graph topology — they only see their own src and dst tensors.

#### Option B: Replace at the matmul activation prep level

Modify the activation preparation in `ggml_compute_forward_mul_mat()` at `ggml.c:13280`. Currently this calls `quantize_row_i8_s()` to convert float32 → int8. Replace this with a fused RMSNorm+quantize that reads from the RMSNorm's float32 INPUT (before normalization) and produces int8 directly.

Problem: the matmul activation prep reads from `src1->data`, which is the output of the PREVIOUS operation (the gamma multiply after RMSNorm). The RMSNorm input is two operations back. We'd need to reach through the graph.

#### Option C: Replace at the graph build level (RECOMMENDED)

Modify `build_bitnet_158()` in `llama.cpp:15389` to NOT create separate RMSNorm + gamma_mul + matmul ops. Instead, create a SINGLE custom op that fuses RMSNorm + gamma + quantize + matmul.

Problem: requires a new ggml op type and compute function. Significant engineering.

#### Option D: Replace at the quantize_row_i8_s call site (SIMPLEST)

The quantize call at `ggml.c:13280` converts `src1->data` (float32) to int8 in `wdata`. The float32 data in `src1->data` has ALREADY been through RMSNorm + gamma. So `quantize_row_i8_s` is converting already-normalized data to int8.

Replace `quantize_row_i8_s` with a version that quantizes to range ±80 instead of ±127. This is what the `SHIRLEY_ACT_RANGE` compile flag already does.

This doesn't fuse anything — it just changes the quantization range. But it's the simplest path to getting 5-trit activations through the real matmul.

The FUSED version (skipping the float32 RMSNorm + gamma intermediate) requires Option C — a custom ggml op. That's the real bandwidth win.

### Recommended path

1. **First: Option D** — change `SHIRLEY_ACT_RANGE` to 80 and verify the full pipeline works with 5-trit activations. We already tested this (PPL fine, generation good at RANGE=80). But now we test it COMBINED with the struct extension and bridge infrastructure to make sure nothing interacts badly. This is a compile flag change, not a code change.

2. **Second: Build the fused op (Option C)** — create `GGML_OP_SHIRLEY_RMSNORM_QUANTIZE` that takes float32 input, gamma weights, and produces int8 + scale in one pass. Register it in the ggml dispatch table. Use it in `build_bitnet_158()` for the 4 RMSNorms that feed ternary matmuls (attn_norm, ffn_norm — the ones BEFORE QKV and gate/up).

3. **Third: Wire the FFN trivials** — once the fused RMSNorm outputs int8, the gate and up matmul outputs can flow through `shirley_relu_i8`, `shirley_square_i8_to_i16`, `shirley_mul_i16_i8_to_i8` in int8. This requires the post-matmul rescale to produce int8 (which is the `shirley_rescale_i32_to_i8` kernel we already built).

4. **Fourth: Measure** — run the full test suite (PPL, generation quality across 4 prompts, throughput) and compare against baseline. The bandwidth savings should show up as faster prompt eval (batch processing moves less data through cache).

## File Map

| File | What's there | What changes |
|------|-------------|-------------|
| `ggml.h` | ggml_tensor with shirley fields | Add GGML_OP_SHIRLEY_RMSNORM_Q (for Option C) |
| `ggml.c` | Dispatch table, compute functions | Add compute_forward for the fused op |
| `llama.cpp` | build_bitnet_158() graph builder | Replace rms_norm+mul with fused op for ternary matmuls |
| `ggml-quants.c` | quantize_row_i8_s | SHIRLEY_ACT_RANGE already works, no change needed |
| `shirley_kernels.h` | All Shirley kernels | Used by the fused op's compute function |
| `shirley_mtfp21.h` | MTFP21 arithmetic | Used by shirley_rmsnorm_ternary for integer rsqrt |

## Test Protocol

After EACH change:

```bash
# 1. Build
cd bitnet && cmake -B build ... && cmake --build build --config Release

# 2. Quick sanity (generation)
builds/test.sh shirley "The meaning of life is" --tokens 64

# 3. Perplexity (if sanity passes)
builds/shirley/perplexity.sh -m models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf \
    -f eval_data/wikitext2_test.txt -t 6 --chunks 20 --ctx-size 512

# 4. A/B generation quality (if PPL passes)
builds/test.sh baseline "Hypothetically, might reflective recursion be a function of cognition?"
builds/test.sh shirley  "Hypothetically, might reflective recursion be a function of cognition?"

# 5. Throughput comparison
# Compare prompt eval and generation tok/s between baseline and shirley
```

PPL acceptance: within ±0.1 of baseline (18.852).
Generation acceptance: coherent, no loops, lexical diversity > 40%.
Throughput acceptance: no regression > 5% on any metric.

## What We Learned the Hard Way

1. **Simulation lies.** The 4-trit quantize-dequantize simulation said PPL was fine. Real integer quantization at 4-trit causes generation loops. Test with REAL data, not simulations.

2. **PPL is not enough.** PPL measures top-1 token prediction. Generation quality depends on distribution shape. Always test generation with diverse prompts.

3. **The baseline is already fast.** ggml's float32 RMSNorm uses SIMD-vectorized `ggml_vec_scale_f32` with loop unrolling. Don't replace fast code with different fast code — replace it with fundamentally different computation (int8 instead of float32).

4. **Struct extensions have cache costs.** Adding 16 bytes to ggml_tensor shifts cache alignment for every tensor in the model. ~5% overhead. This is the tax for Path B.

5. **The model uses ReLU-squared, not SiLU.** Read the actual forward pass code before making assumptions.

6. **RMSNorm IS the bridge.** The domain transition from float32 to int8 happens at RMSNorm, not at the matmul post-processing. The matmul side-buffer infrastructure is useful but the actual bridge is the fused RMSNorm+quantize.

7. **Per-block quantization scales don't work.** A single tensor needs a single scale. Multi-threaded code that writes per-block scales to a single float is a race condition AND produces garbage data.

8. **Test before committing.** Don't ship scaffolding as progress. If it's not tested end-to-end, it's not done.
