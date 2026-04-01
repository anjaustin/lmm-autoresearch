# BitNet b1.58-2B-4T: End-to-End Ternary Integration Map

## Purpose

This document maps every atomic operation in one forward pass of BitNet b1.58-2B-4T, identifies the current data format, the ternary/MTFP21 target format, what we've built, and what remains. It is the execution plan for end-to-end ternary inference.

## Architecture Summary

**Model:** Microsoft BitNet b1.58-2B-4T (2.41B params, 30 layers, dim 2560, 20 heads, 5 KV heads, GQA=4)

**Current pipeline:** Ternary weights (2-bit {0,1,2} packed) with float32 activations. The matmul kernel operates in integer (int8 activations × 2-bit weights via AVX2), but everything between matmuls is float32.

**Target pipeline:** Integer activations (int8, ~5-trit range ±80) flowing through integer operations (AVX2) with MTFP21 for scalar bookkeeping (scale factors, normalization parameters). Float only for operations that are inherently float (attention score computation, RoPE, softmax, embedding/LM-head).

**Three-layer compute stack:**
```
Layer 1: AVX2 integer    Per-element bulk (sign_epi8, mullo, mulhrs, max, add)
Layer 2: Scalar FPU      One-off transcendentals (rsqrt via MTFP21 LUT+NR or sqrtf)
Layer 3: MTFP21          Transport format for scale factors between stages
```

## The Forward Pass: Every Atomic Operation

### Notation

| Symbol | Meaning |
|--------|---------|
| f32 | float32 tensor |
| i8 | int8 tensor (range ±SHIRLEY_ACT_RANGE, default 80) |
| i16 | int16 intermediate |
| i32 | int32 accumulator |
| 2bit | 2-bit packed ternary weights {0,1,2} → {-1,0,+1} |
| MTFP | MTFP21 scalar (mantissa int32 + exponent int8) |
| γ | learned per-element weight (gamma) |
| s | learned per-layer scalar scale factor |

### Embedding (once, before layer 0)

```
E1: Token embedding lookup
    Current:  token_id (int32) → tok_embd table → float32 vector [2560]
    Ternary:  KEEP FLOAT — embedding table is float32, not ternary.
              Quantize output to int8 after lookup for first layer input.
    Status:   Need quantize step after embedding
    Priority: LOW — happens once per token, not per layer
```

### Per-Layer Operations (×30 layers)

#### Attention Block

```
A1: RMSNorm (attn_norm)
    Current:  f32 [2560] → ggml_rms_norm → f32 [2560], then ggml_mul(cur, γ)
    Ternary:  i8 [2560] → shirley_rmsnorm_ternary → i8 [2560] (with Q14 gamma)
    Status:   KERNEL BUILT (417 ns, zero float, 31/31 tests)
    Integration: Replace ggml_compute_forward_rms_norm_f32 + ggml_mul
    Needs:    Load model gamma weights into Q14 int16 format at model load time
    File:     ggml.c:12097 (rms_norm_f32), llama.cpp:9512 (gamma mul)
```

```
A2: Activation quantization (before matmul)
    Current:  f32 [2560] → quantize_row_i8_s → i8 [2560] + act_scale (float)
    Ternary:  i8 [2560] already int8 from A1 — NO QUANTIZATION NEEDED
              The act_scale becomes part of the MTFP21 bookkeeping.
              If A1 outputs int8 at range ±80, act_scale = 1.0 (identity).
    Status:   ELIMINATION — this step disappears when RMSNorm outputs int8
    Integration: Skip quantize_row_i8_s when input is already int8
    File:     ggml.c:13166 (quantize call site)
```

```
A3-A5: Ternary matmul (wq, wk, wv × normalized input)
    Current:  i8 [2560] × 2bit [N×2560] → int32 → f32 (via act_scale/wt_scale)
    Ternary:  i8 [2560] × 2bit [N×2560] → int32 → i8 (via fixed-point rescale)
    Status:   MATMUL DONE (sign_epi8 validated). POST-PROCESSING NEEDS CHANGE.
    Integration: Modify the post-matmul rescale (see A6)
    File:     ggml.c:12513-12521 (post-processing), ggml-bitnet-mad.cpp (kernel)
```

```
A6: Post-matmul rescale (THE CRITICAL INTEGRATION POINT)
    Current:  raw_int32 = dot_product result
              float_result = (raw_int32 / act_scale) * weight_scale
              This produces float32 output that feeds RoPE, attention, etc.

    Ternary:  raw_int32 = dot_product result
              Need to produce int8 output with an MTFP21 scale factor.
              int8_result = clamp(round(raw_int32 * combined_scale), ±80)
              where combined_scale = weight_scale / act_scale (precomputable)

              The combined_scale is a per-layer constant (depends only on
              weight_scale and the input quantization range). It can be
              precomputed at model load time as an MTFP21 value, then
              converted to Q15 fixed-point for the AVX2 scale multiply.

    Status:   NEEDS BUILDING — this is the bridge between integer matmul
              and integer RMSNorm. Currently forces conversion to float32.
    Integration: Modify ggml.c:12513 to output int8 instead of float32
    File:     ggml.c:12504-12526, ggml.c:13269-13290
    Complexity: MEDIUM — same pattern as shirley_rmsnorm_ternary Phase 5
                (Q15 multiply + shift + pack)
```

```
A7-A8: RoPE (Rotary Position Encoding) on Q and K
    Current:  f32 Q/K [n_head × n_embd_head × n_tokens] → apply sin/cos → f32
    Ternary:  KEEP FLOAT — RoPE requires sin/cos (transcendental).
              The Q and K tensors enter the attention mechanism in float32.
              This is acceptable because attention (A9-A11) is already float.
              Alternative: precompute sin/cos tables as fixed-point and apply
              in integer. Low priority — attention is float anyway.
    Status:   KEEP FLOAT (acceptable — feeds into float attention)
    Priority: LOW
    File:     llama.cpp:9447-9458 (ggml_rope_ext calls)
```

```
A9: Attention scores (Q @ K^T)
    Current:  f32 Q × f32 K^T → f32 scores
    Ternary:  KEEP FLOAT — both operands are float32 (post-RoPE).
              This is NOT a ternary matmul. Weights are not ternary.
    Status:   KEEP FLOAT (inherently float×float)
    File:     llama.cpp:9815 (ggml_mul_mat(k, q))
```

```
A10: Softmax
    Current:  f32 scores → exp(x) / Σexp(x) → f32 attention weights
    Ternary:  KEEP FLOAT for now.
              Future: iGPU candidate (V_EXP_F32 hardware unit).
              156M EXP operations per forward pass across all layers/heads.
              This is where the iGPU earns its place — not in RMSNorm.
    Status:   KEEP FLOAT (iGPU integration is a separate project)
    File:     llama.cpp:9844 (ggml_soft_max_ext)
```

```
A11: Attention-weighted value sum (attn_weights @ V)
    Current:  f32 attn_weights × f32 V → f32 context
    Ternary:  KEEP FLOAT — same as A9, both operands are float32.
    Status:   KEEP FLOAT
    File:     llama.cpp:9858 (ggml_mul_mat(v, kq))
```

```
A12: RMSNorm (attn_sub_norm) — normalizes attention output before wo projection
    Current:  f32 → rms_norm → f32, then gamma mul
    Ternary:  f32 → shirley_rmsnorm_quantize → i8
              (This RMSNorm takes FLOAT input from attention and produces
              INT8 output for the next ternary matmul. Uses the f32→i8
              fused variant, not the i8→i8 ternary variant.)
    Status:   KERNEL BUILT (shirley_rmsnorm_quantize, 9.8 us — needs SIMD)
    Integration: Replace rms_norm + gamma + quantize with fused kernel
    File:     llama.cpp:15465-15468
```

```
A13: Activation quantization (before wo matmul)
    Current:  f32 → quantize_row_i8_s → i8
    Ternary:  ELIMINATED — fused into A12 (shirley_rmsnorm_quantize)
    Status:   ELIMINATION
```

```
A14: Ternary matmul (wo × attn_sub_norm output)
    Current:  i8 × 2bit → int32 → f32
    Ternary:  i8 × 2bit → int32 → i8 (via A6-style rescale)
    Status:   MATMUL DONE, POST-PROCESSING NEEDS CHANGE (same as A6)
```

```
A15: wo_scale multiply
    Current:  f32 × scalar → f32
    Ternary:  Fold into the post-matmul rescale (A6).
              combined_scale = weight_scale * wo_scale / act_scale
              This is a per-layer constant, precomputed at model load.
    Status:   FOLD INTO A6
```

```
A16: Residual ADD (attention)
    Current:  f32 + f32 → f32 (cur + inpSA)
    Ternary:  i8 + i8 → i8 (or i16 if range exceeds ±127)
              For two i8 values in ±80, sum is in ±160.
              Fits i8 only if we allow the residual to briefly use wider range.
              Or: accumulate in i16, requantize to i8 before next RMSNorm.
    Status:   NEEDS BUILDING — trivial AVX2 (add_epi8 or add_epi16 + pack)
    Complexity: LOW — but the range management needs thought
    File:     llama.cpp:15488 (ggml_add)
```

#### FFN Block

```
F1: RMSNorm (ffn_norm)
    Current:  f32 → rms_norm + gamma → f32
    Ternary:  i8 → shirley_rmsnorm_ternary (with Q14 gamma) → i8
              (Input is int8 from residual. Output feeds two parallel matmuls.)
    Status:   KERNEL BUILT
    File:     llama.cpp:15491-15494
```

```
F2: Activation quantization (before gate and up matmuls)
    Current:  f32 → quantize_row_i8_s → i8
    Ternary:  ELIMINATED — F1 already outputs int8
    Status:   ELIMINATION
```

```
F3-F4: Ternary matmul (gate × norm, up × norm) — PARALLEL
    Current:  i8 × 2bit → int32 → f32
    Ternary:  i8 × 2bit → int32 → i8 (via rescale)
    Note:     LLM_FFN_PAR means gate uses the ORIGINAL input (cur),
              up uses the ORIGINAL input (cur). Both read the same i8 vector.
    Status:   MATMUL DONE, POST-PROCESSING NEEDS CHANGE
    File:     llama.cpp:9564-9568 (PAR gate), 9544 (up)
```

```
F5-F6: gate_scale and up_scale multiplies
    Current:  f32 × scalar → f32
    Ternary:  Fold into post-matmul rescale (same pattern as A15).
    Status:   FOLD INTO RESCALE
```

```
F7: ReLU on gate
    Current:  f32 → max(0, x) → f32
    Ternary:  i8 → max(0, x) → i8
              AVX2: _mm256_max_epi8(x, _mm256_setzero_si256())
              ONE INSTRUCTION. This is the MAX prime — CPU-native.
    Status:   NEEDS BUILDING (trivial — one intrinsic)
    Complexity: TRIVIAL
    File:     llama.cpp:9607 (ggml_relu)
```

```
F8: Square(ReLU(gate))
    Current:  f32 → x² → f32
    Ternary:  i8 → x² → i16
              AVX2: widen to i16, _mm256_mullo_epi16(x, x)
              After ReLU, values are in [0, 80]. Squares in [0, 6400]. Fits i16.
    Status:   NEEDS BUILDING (trivial — widen + one intrinsic)
    Complexity: TRIVIAL
    File:     llama.cpp:9610 (ggml_sqr)
```

```
F9: gate² × up (element-wise multiply)
    Current:  f32 × f32 → f32
    Ternary:  i16 (gate²) × i8 (up) → i16 or i32
              After F8: gate² is in [0, 6400] (i16).
              up is in [-80, 80] (i8 from post-matmul rescale).
              Product: [-512000, +512000]. Fits i32, NOT i16.
              Need: widen up to i16, multiply i16×i16→i32, then requantize to i8.
              Or: keep in i16 with a scale factor tracked in MTFP21.
    Status:   NEEDS BUILDING
    Complexity: MEDIUM — the range management is the challenge
    File:     llama.cpp:9629 (ggml_mul, gate_par)
```

```
F10: RMSNorm (ffn_sub_norm)
    Current:  f32 → rms_norm + gamma → f32
    Ternary:  i8 (or i16 from F9, requantized) → shirley_rmsnorm_ternary → i8
    Status:   KERNEL BUILT (but input format depends on F9 output)
    File:     llama.cpp:15504-15507
```

```
F11: Activation quantization (before down matmul)
    Current:  f32 → quantize_row_i8_s → i8
    Ternary:  ELIMINATED — F10 outputs int8
    Status:   ELIMINATION
```

```
F12: Ternary matmul (down × ffn_sub_norm)
    Current:  i8 × 2bit → int32 → f32
    Ternary:  i8 × 2bit → int32 → i8 (via rescale)
    Status:   MATMUL DONE, POST-PROCESSING NEEDS CHANGE
```

```
F13: down_scale multiply
    Current:  f32 × scalar → f32
    Ternary:  Fold into post-matmul rescale.
    Status:   FOLD INTO RESCALE
```

```
F14: Residual ADD (FFN)
    Current:  f32 + f32 → f32 (cur + ffn_inp)
    Ternary:  i8 + i8 → i8 (same range issue as A16)
    Status:   NEEDS BUILDING (same as A16)
```

### Output (once, after layer 29)

```
O1: RMSNorm (output_norm)
    Current:  f32 → rms_norm + gamma → f32
    Ternary:  i8 → shirley_rmsnorm_ternary → i8 (or f32 for LM head)
              The LM head (O2) uses float32 embedding weights, so the
              output of this RMSNorm may need to be float32 regardless.
              Use shirley_rmsnorm_f32 for this one instance.
    Status:   KERNEL BUILT (shirley_rmsnorm_f32 for float output)
    File:     llama.cpp:15524-15526
```

```
O2: LM head matmul
    Current:  f32 × f32 (tok_embd) → f32 logits
    Ternary:  KEEP FLOAT — tok_embd is float32, not ternary.
              The LM head reuses the embedding table (model.tok_embd).
              This is a standard float matmul, not a ternary matmul.
    Status:   KEEP FLOAT
    File:     llama.cpp:15530
```

## Integration Scorecard

| Category | Count | Status | Remaining work |
|----------|-------|--------|---------------|
| Ternary matmul (sign_epi8) | 7 | **DONE** | 0 |
| RMSNorm kernel | 5 | **BUILT** | Integration into ggml dispatch |
| Activation quantize | 4 | **ELIMINATED** | Remove when RMSNorm outputs int8 |
| Post-matmul rescale (A6) | 7 | **NEEDS BUILDING** | Fixed-point rescale, same as RMSNorm Phase 5 |
| Scale factor fold-in | 4 | **NEEDS BUILDING** | Fold wo/gate/up/down_scale into rescale |
| ReLU | 1 | **NEEDS BUILDING** | Trivial: one AVX2 instruction |
| Square | 1 | **NEEDS BUILDING** | Trivial: widen + one AVX2 instruction |
| Element-wise MUL (F9) | 1 | **NEEDS BUILDING** | Medium: range management |
| Residual ADD | 2 | **NEEDS BUILDING** | Low: i8 add with overflow to i16 |
| RoPE | 2 | **KEEP FLOAT** | Feeds into float attention |
| Attention matmul | 2 | **KEEP FLOAT** | Inherently float×float |
| Softmax | 1 | **KEEP FLOAT** | iGPU candidate (future) |
| Embedding | 1 | **KEEP FLOAT** | Float lookup + quantize |
| LM head | 1 | **KEEP FLOAT** | Float matmul (not ternary) |

**Total: 39 operations. 16 done/eliminated. 16 need building. 7 stay float.**

## The Critical Path

The operations that need building, in dependency order:

### Phase 1: Post-matmul rescale (A6) — THE BRIDGE

This is the single most important integration point. Every ternary matmul currently outputs float32 via `(raw_int32 / act_scale) * weight_scale`. For end-to-end ternary, it needs to output int8 with the scale tracked in MTFP21.

The pattern is identical to `shirley_rmsnorm_ternary` Phase 5:
- Compute combined_scale = weight_scale / act_scale (precompute at model load)
- Convert to Q15 fixed-point
- `result_i8 = clamp(mulhrs(raw_result_i16, combined_scale_q15), ±out_range)`

If act_scale is eliminated (because RMSNorm already outputs int8 at range ±80), the combined_scale simplifies to just weight_scale, which is a per-layer constant.

**This unblocks everything downstream.** Once the matmul outputs int8, the RMSNorm kernel can consume it directly, the residual ADD works in int8, and the activation quantize step is eliminated.

### Phase 2: Trivial integer operations

- **ReLU:** `_mm256_max_epi8(x, zero)` — one instruction
- **Square:** widen to i16, `_mm256_mullo_epi16(x, x)` — two instructions
- **Residual ADD:** `_mm256_add_epi8(a, b)` or `_mm256_add_epi16(a, b)` — one instruction

These are CPU-native primes (MAX, MUL, ADD). No engineering challenge.

### Phase 3: Element-wise MUL with range management (F9)

gate² (i16, range [0, 6400]) × up (i8, range [-80, +80]) → product (i32, range [-512000, +512000]).

Need to requantize to int8 before the next RMSNorm. This requires:
1. Find max absolute value of the product vector
2. Compute quantization scale = 80 / max_abs
3. Apply: `result_i8 = clamp(round(product * quant_scale), ±80)`

Same pattern as the post-matmul rescale (Phase 1). The scale factor goes into MTFP21 for downstream bookkeeping.

### Phase 4: RMSNorm integration

Wire `shirley_rmsnorm_ternary` (or `shirley_rmsnorm_f32` for the output norm) into the ggml dispatch. Replace `ggml_compute_forward_rms_norm_f32` at the 5 call sites. Load gamma weights into Q14 format at model load time.

### Phase 5: Scale factor plumbing

The wo_scale, gate_scale, up_scale, and down_scale are per-layer learned scalars. Currently applied as `ggml_mul(cur, scale_tensor)` in float32. For ternary: fold these into the post-matmul rescale's combined_scale constant. This is a model-load-time precomputation, not a runtime operation.

## Data Format Transitions

In the end-to-end ternary pipeline, data flows like this:

```
Embedding (f32) → [quantize] → i8

Per layer:
  i8 → RMSNorm → i8 → matmul(QKV) → int32 → [rescale] → i8/f32

  For Q,K: i8 → [dequant to f32] → RoPE → f32
  Attention: f32 → Q@K^T → softmax → attn@V → f32

  f32 → RMSNorm_sub → [fused quantize] → i8
  i8 → matmul(wo) → int32 → [rescale] → i8
  i8 + i8 → [residual] → i8

  i8 → RMSNorm → i8 → matmul(gate,up) → int32 → [rescale] → i8
  i8 → ReLU → i8 → Square → i16
  i16 × i8 → [rescale] → i8

  i8 → RMSNorm_sub → i8 → matmul(down) → int32 → [rescale] → i8
  i8 + i8 → [residual] → i8

Output:
  i8 → RMSNorm → f32 → LM_head(f32) → f32 logits → sampling
```

The format transitions are:
- **i8 ↔ int32:** at every ternary matmul (the rescale bridge)
- **i8 ↔ i16:** at ReLU-squared (widen for square, requantize after)
- **i8 → f32:** at the attention boundary (Q,K need float for RoPE)
- **f32 → i8:** at RMSNorm_sub after attention (fused norm+quantize)
- **i8 → f32:** at the output norm (LM head needs float)

Every transition has an MTFP21 scale factor that tracks the quantization.

## What Stays Float (and Why)

| Operation | Why float | Could it go ternary? |
|-----------|----------|---------------------|
| RoPE | sin/cos transcendental | Yes, with precomputed LUT. Low priority. |
| Q @ K^T | float×float operands | No — Q,K are float after RoPE |
| Softmax | exp() transcendental | iGPU candidate (V_EXP_F32). Future project. |
| attn @ V | float×float operands | No — attention weights are float |
| Embedding | float32 table | Could quantize table to int8. Low priority. |
| LM head | float32 weights | Reuses embedding table. Same as above. |
| Output RMSNorm→f32 | LM head needs float | Last operation before float logits. Acceptable. |

The float operations are concentrated in the attention mechanism (RoPE + Q@K^T + softmax + attn@V) and the model boundaries (embedding, LM head). Everything between the attention outputs and the next attention inputs is ternary integer.

## Scale Factor Bookkeeping

Every int8 tensor in the pipeline carries an implicit scale factor:
```
real_value = int8_value / scale_factor
```

The MTFP21 transport layer tracks these scales:

| Scale | Source | When computed | How used |
|-------|--------|--------------|----------|
| act_scale | Activation quantization | Per forward pass | Divides matmul result |
| weight_scale | Model weights | Model load | Multiplies matmul result |
| wo/gate/up/down_scale | Model weights | Model load | Folds into rescale |
| norm_scale | RMSNorm rsqrt | Per RMSNorm call | Applied via Q15 multiply |
| quant_scale | Post-matmul rescale | Per matmul call | Requantizes int32→int8 |

These are all scalars (not per-element). They can be precomputed and combined to minimize the number of runtime multiply operations.

## Origin

Integration map created April 1, 2026 from Session 002 architecture audit of `build_bitnet_158()` in `llama.cpp:15389-15537`, `llm_build_ffn()` in `llama.cpp:9526-9651`, and `llm_build_kqv()` in `llama.cpp:9754-9882`. Every operation verified against the actual model forward pass code.
