# BitNet b1.58-2B-4T: End-to-End Ternary Integration Map

## Purpose

This document maps every atomic operation in one forward pass of BitNet b1.58-2B-4T. Every operation is mapped to its integer/MTFP21/iGPU implementation. There is no float32 in this pipeline. Float32 is a legacy format — the Shirley pipeline replaces it entirely with:

- **AVX2 integer (Layer 1):** Per-element bulk compute — sign_epi8, mullo_epi16, mulhrs_epi16, add, max
- **MTFP21 (Layer 3):** Scale factor bookkeeping, precision transport between stages
- **iGPU transcendental (Layer 2):** EXP and LOG via hardware units (V_EXP_F32, V_LOG_F32)

The six primes on two substrates. That is the architecture.

## Architecture Summary

**Model:** Microsoft BitNet b1.58-2B-4T (2.41B params, 30 layers, dim 2560, 20 heads, 5 KV heads, GQA=4)

**Current pipeline:** Ternary weights (2-bit {0,1,2} packed) with float32 activations. The matmul kernel operates in integer (int8 activations × 2-bit weights via AVX2), but everything between matmuls is float32.

**Target pipeline:** Integer activations (int8, range ±80) flowing through integer operations (AVX2) with MTFP21 for scale tracking. EXP/LOG routed to iGPU hardware transcendental units via shared memory. No float32 anywhere in the compute path.

**Three-layer compute stack:**
```
Layer 1: AVX2 integer    Per-element bulk (sign_epi8, mullo, mulhrs, max, add)
                         32 elements/cycle, per-element work
                         Primes: ADD, MUL, MAX, CONST

Layer 2: iGPU (RDNA)     Transcendental operations (V_EXP_F32, V_LOG_F32)
                         Hardware units, ~4 cycles per op
                         Primes: EXP, LOG
                         Shared memory — no PCIe transfer

Layer 3: MTFP21          Scale factors, normalization parameters, precision transport
                         25.4-bit precision (exceeds float32)
                         Integer-only: mantissa (int32) × 3^exponent (int8)
```

## Notation

| Symbol | Meaning |
|--------|---------|
| i8 | int8 tensor (range ±80, 5-trit) |
| i16 | int16 intermediate |
| i32 | int32 accumulator |
| 2bit | 2-bit packed ternary weights {0,1,2} → {-1,0,+1} |
| MTFP | MTFP21 scalar (mantissa int32 + exponent int8) |
| γ | learned per-element weight (gamma), stored as Q14 int16 |
| s | learned per-layer scalar scale factor, stored as MTFP21 |
| iGPU | operation routed to integrated GPU transcendental unit |

## The Forward Pass: Every Atomic Operation

### Embedding (once, before layer 0)

```
E1: Token embedding lookup
    Current:  token_id (int32) → tok_embd table → float32 vector [2560]
    Target:   token_id (int32) → tok_embd table (int8 + MTFP21 scale) → i8 [2560]
              Embedding table quantized to int8 at model load time.
              Each row stored as int8 with a per-row MTFP21 scale factor.
              Lookup returns int8 directly — no conversion needed.
    Status:   NEEDS BUILDING
    Priority: MEDIUM — happens once per token, but sets the format for the entire pipeline
    Needs:    Embedding table quantization at model load (one-time)
```

### Per-Layer Operations (×30 layers)

#### Attention Block

```
A1: RMSNorm (attn_norm)
    Current:  f32 [2560] → ggml_rms_norm → f32 [2560], then ggml_mul(cur, γ)
    Target:   i8 [2560] → shirley_rmsnorm_ternary → i8 [2560]
              Uses MTFP21 integer rsqrt (LUT + 2 Newton-Raphson).
              Gamma applied via Q14 fixed-point multiply.
              Zero float operations.
    Status:   KERNEL BUILT (417 ns, zero float, 31/31 tests)
    Needs:    Load gamma weights into Q14 int16 format at model load time.
              Wire into ggml dispatch replacing ggml_compute_forward_rms_norm_f32.
    File:     shirley_kernels.h (shirley_rmsnorm_ternary)
```

```
A2: Activation quantization (before matmul)
    Current:  f32 [2560] → quantize_row_i8_s → i8 [2560] + act_scale (float)
    Target:   ELIMINATED — A1 already outputs int8. No quantization step needed.
              The MTFP21 scale from A1 carries forward as the activation scale.
    Status:   ELIMINATION
```

```
A3-A5: Ternary matmul (wq, wk, wv × normalized input)
    Current:  i8 [2560] × 2bit [N×2560] → int32 → f32 (via act_scale/wt_scale)
    Target:   i8 [2560] × 2bit [N×2560] → int32
              Matmul stays exactly as-is — sign_epi8 ternary multiply, integer
              accumulation to int32. The post-processing changes (see A6).
    Status:   MATMUL DONE (sign_epi8 validated, PPL identical to baseline)
    File:     ggml-bitnet-mad.cpp
```

```
A6: Post-matmul rescale (int32 → int8) — THE BRIDGE
    Current:  raw_int32 → float_result = (raw / act_scale) * weight_scale → f32
    Target:   raw_int32 → shirley_rescale_i32_to_i8 → i8 [N] + MTFP21 scale
              combined_scale = weight_scale / act_scale (precomputed at model load
              as MTFP21, converted to Q15 for the AVX2 multiply).
              Output is int8 at range ±80 with MTFP21 scale factor.
              For wq/wk: output feeds RoPE (A7-A8) — stays int8, RoPE applied in integer.
              For wv: output feeds attention weighted sum (A11) — stays int8.
    Status:   KERNEL BUILT (shirley_rescale_i32_to_i8, shirley_rescale_i32_scaled_to_i8)
    Needs:    Wire into ggml.c post-matmul code, replacing float rescale.
              Precompute combined_scale as MTFP21 at model load.
    File:     shirley_kernels.h, ggml.c:12658-12694 (vec_dot paths),
              ggml.c:13445-13496 (gemm/gemv paths)
```

```
A7-A8: RoPE (Rotary Position Encoding) on Q and K
    Current:  f32 Q/K → apply sin/cos → f32
    Target:   i8 Q/K → apply precomputed sin/cos (fixed-point int16) → i8
              Sin/cos tables precomputed at model load as Q15 int16.
              RoPE is: q_rot = q * cos + q_rotated * sin
              In integer: Q15 multiply (mulhrs_epi16) + add_epi16, pack back to i8.
              This is MUL + ADD — two CPU-native primes. No transcendental needed
              at runtime because sin/cos are precomputed constants (CONST prime).
    Status:   NEEDS BUILDING
    Needs:    Precompute sin/cos as Q15 int16 tables at model load.
              Build shirley_rope_i8() kernel — two mulhrs + add + pack per element.
    Priority: HIGH — on the critical path for attention
```

```
A9: Attention scores (Q @ K^T)
    Current:  f32 Q × f32 K^T → f32 scores
    Target:   i8 Q × i8 K^T → i32 scores + MTFP21 combined scale
              This is a standard integer matmul, not a ternary matmul.
              Both operands are int8 (from post-RoPE). Use AVX2 integer
              dot product: _mm256_maddubs_epi16 + _mm256_madd_epi16.
              Scale factor = Q_scale * K_scale (MTFP21 multiply, one scalar op).
              Output: int32 per attention position with MTFP21 scale.
    Status:   NEEDS BUILDING
    Needs:    Build shirley_matmul_i8_i8() — int8×int8 → int32 with MTFP21 scale.
              Dimensions: [n_tokens, head_dim] × [head_dim, n_tokens] per head.
    Priority: HIGH
```

```
A10: Softmax
    Current:  f32 scores → exp(x) / Σexp(x) → f32 attention weights
    Target:   i32 scores → iGPU: exp(x) → i32 numerators → CPU: sum + rescale → i8 weights
              This is where the iGPU earns its place.
              1. Subtract max (MAX prime, CPU): stabilize for exp
              2. Route to iGPU: V_EXP_F32 hardware unit computes exp for all positions
                 (shared memory — no PCIe transfer on APU)
              3. Return to CPU: sum the exponentials (ADD prime)
              4. Normalize: divide each by sum (MUL prime with MTFP21 reciprocal)
              5. Quantize result to int8 for attn@V matmul
              156M EXP operations per forward pass across all layers/heads.
              iGPU V_EXP_F32: ~4 cycles per op, massively parallel CUs.
    Status:   NEEDS BUILDING
    Needs:    Build MTFP21 exp() — LUT + polynomial correction (same pattern as rsqrt).
              Build iGPU dispatch for EXP via shared memory.
              Build shirley_softmax() orchestrator — CPU subtract max, iGPU exp,
              CPU normalize + quantize.
              Fallback: CPU software exp via MTFP21 LUT until iGPU dispatch is wired.
    Priority: HIGH — but can start with CPU MTFP21 fallback, add iGPU later
```

```
A11: Attention-weighted value sum (attn_weights @ V)
    Current:  f32 attn_weights × f32 V → f32 context
    Target:   i8 attn_weights × i8 V → i32 context + MTFP21 scale
              Same as A9 — standard integer matmul.
              attn_weights are int8 (from softmax quantization).
              V is int8 (from post-matmul rescale A6).
              Output: int32 per head dimension with MTFP21 scale.
              Rescale to int8 for downstream.
    Status:   NEEDS BUILDING
    Needs:    Same shirley_matmul_i8_i8() kernel as A9.
    Priority: HIGH
```

```
A12: RMSNorm (attn_sub_norm)
    Current:  f32 → rms_norm → f32, then gamma mul
    Target:   i8 → shirley_rmsnorm_ternary → i8
              (Same kernel as A1. Input is int8 from attention output rescale.)
    Status:   KERNEL BUILT
    File:     shirley_kernels.h (shirley_rmsnorm_ternary)
```

```
A13: Activation quantization (before wo matmul)
    Current:  f32 → quantize_row_i8_s → i8
    Target:   ELIMINATED — A12 already outputs int8
    Status:   ELIMINATION
```

```
A14: Ternary matmul (wo × attn_sub_norm output)
    Current:  i8 × 2bit → int32 → f32
    Target:   i8 × 2bit → int32 → i8 (via A6-style rescale)
    Status:   MATMUL DONE, POST-PROCESSING NEEDS CHANGE (same as A6)
```

```
A15: wo_scale multiply
    Current:  f32 × scalar → f32
    Target:   Fold into the post-matmul rescale (A6).
              combined_scale = weight_scale * wo_scale / act_scale
              Precomputed at model load as MTFP21.
    Status:   FOLD INTO A6
```

```
A16: Residual ADD (attention)
    Current:  f32 + f32 → f32 (cur + inpSA)
    Target:   i8 + i8 → i16 → i8
              For two i8 values in ±80, sum is in ±160.
              Accumulate in i16 via _mm256_cvtepi8_epi16 + _mm256_add_epi16.
              Requantize to i8 via shirley_requantize_i16_to_i8.
              MTFP21 scale factor updated to track the combined scale.
    Status:   KERNEL BUILT (shirley_residual_add_i16)
    Needs:    Wire into pipeline. Scale factor bookkeeping.
    Complexity: LOW
```

#### FFN Block

```
F1: RMSNorm (ffn_norm)
    Current:  f32 → rms_norm + gamma → f32
    Target:   i8 → shirley_rmsnorm_ternary (with Q14 gamma) → i8
    Status:   KERNEL BUILT
```

```
F2: Activation quantization (before gate and up matmuls)
    Current:  f32 → quantize_row_i8_s → i8
    Target:   ELIMINATED — F1 already outputs int8
    Status:   ELIMINATION
```

```
F3-F4: Ternary matmul (gate × norm, up × norm) — PARALLEL
    Current:  i8 × 2bit → int32 → f32
    Target:   i8 × 2bit → int32 → i8 (via A6-style rescale)
    Status:   MATMUL DONE, POST-PROCESSING NEEDS CHANGE
```

```
F5-F6: gate_scale and up_scale multiplies
    Current:  f32 × scalar → f32
    Target:   Fold into post-matmul rescale (same pattern as A15).
    Status:   FOLD INTO RESCALE
```

```
F7: ReLU on gate
    Current:  f32 → max(0, x) → f32
    Target:   i8 → max(0, x) → i8
              AVX2: _mm256_max_epi8(x, _mm256_setzero_si256())
              ONE INSTRUCTION. This is the MAX prime — CPU-native.
    Status:   KERNEL BUILT (shirley_relu_i8)
    File:     shirley_kernels.h
```

```
F8: Square(ReLU(gate))
    Current:  f32 → x² → f32
    Target:   i8 → x² → i16
              AVX2: widen to i16, _mm256_mullo_epi16(x, x)
              After ReLU, values are in [0, 80]. Squares in [0, 6400]. Fits i16.
    Status:   KERNEL BUILT (shirley_square_i8_to_i16)
    File:     shirley_kernels.h
```

```
F9: gate² × up (element-wise multiply)
    Current:  f32 × f32 → f32
    Target:   i16 (gate²) × i8 (up) → i32 → i8
              gate² is in [0, 6400] (i16). up is in [-80, 80] (i8).
              Product: [-512000, +512000]. Fits i32.
              Widen both to i16, multiply i16×i16→i32, then
              shirley_rescale_i32_to_i8 → i8 at ±80 with MTFP21 scale.
    Status:   KERNEL BUILT (shirley_mul_i16_i8_to_i8)
    File:     shirley_kernels.h
```

```
F10: RMSNorm (ffn_sub_norm)
    Current:  f32 → rms_norm + gamma → f32
    Target:   i8 → shirley_rmsnorm_ternary → i8
    Status:   KERNEL BUILT
```

```
F11: Activation quantization (before down matmul)
    Current:  f32 → quantize_row_i8_s → i8
    Target:   ELIMINATED — F10 outputs int8
    Status:   ELIMINATION
```

```
F12: Ternary matmul (down × ffn_sub_norm)
    Current:  i8 × 2bit → int32 → f32
    Target:   i8 × 2bit → int32 → i8 (via rescale)
    Status:   MATMUL DONE, POST-PROCESSING NEEDS CHANGE
```

```
F13: down_scale multiply
    Current:  f32 × scalar → f32
    Target:   Fold into post-matmul rescale.
    Status:   FOLD INTO RESCALE
```

```
F14: Residual ADD (FFN)
    Current:  f32 + f32 → f32 (cur + ffn_inp)
    Target:   i8 + i8 → i16 → i8 (same as A16)
    Status:   KERNEL BUILT (shirley_residual_add_i16)
```

### Output (once, after layer 29)

```
O1: RMSNorm (output_norm)
    Current:  f32 → rms_norm + gamma → f32
    Target:   i8 → shirley_rmsnorm_ternary → i8
              Output feeds the LM head matmul (O2).
    Status:   KERNEL BUILT
```

```
O2: LM head matmul
    Current:  f32 × f32 (tok_embd) → f32 logits
    Target:   i8 × i8 (quantized tok_embd) → i32 logits + MTFP21 scale
              The embedding table is quantized to int8 at model load (same as E1).
              Standard integer matmul: i8 × i8 → i32.
              Output is int32 logits with MTFP21 scale factor.
    Status:   NEEDS BUILDING
    Needs:    Same shirley_matmul_i8_i8() as A9/A11, or reuse existing ggml
              int8 matmul path. Embedding table quantization shared with E1.
    Priority: MEDIUM
```

```
O3: Sampling
    Current:  f32 logits → softmax → token selection
    Target:   i32 logits → iGPU: softmax (EXP prime) → probability → token selection
              Same softmax as A10. The probabilities are the last computation
              before discrete token selection. MTFP21 carries precision through
              to the final probability comparison.
              This is the ONLY place where a continuous value becomes a discrete
              output — and it's a selection (MAX prime), not a float operation.
    Status:   NEEDS BUILDING (shares implementation with A10)
    Priority: MEDIUM — can reuse A10's softmax
```

## Integration Scorecard

| Category | Count | Status | Remaining work |
|----------|-------|--------|---------------|
| Ternary matmul (sign_epi8) | 7 | **DONE** | 0 |
| Post-matmul rescale (i32→i8) | 7 | **INTEGRATED** | 0 (commit 843eef4) |
| FFN block (MTFP21 custom op) | 1 | **INTEGRATED** | 0 (commit b0489e1) |
| MTFP21 exp + softmax + cmp | 1 | **BUILT** | Wire into attention path |
| RoPE (integer) | 2 | **NEEDS BUILDING** | Precompute sin/cos as Q15, build kernel |
| Integer matmul (i8×i8→i32) | 3 | **NEEDS BUILDING** | Q@K^T, attn@V, LM head |
| Embedding (i8) | 1 | **NEEDS BUILDING** | Quantize table at model load |
| Attention custom op | 1 | **NEEDS BUILDING** | Same pattern as FFN custom op |

**Total: 40 operations. FFN path complete. Attention path remaining.**

## The Critical Path

### Phase 1: Post-matmul rescale — COMPLETE

All 7 ternary matmuls write int8 via `shirley_rescale_raw_to_i8` at all 4 ggml.c code paths (16-row vec_dot, 1-row fallback, gemm, gemv). 5-trit activations (SHIRLEY_ACT_RANGE=80) validated: PPL 18.888, baseline 18.852.

Commits: bc23f57 (Option D validation), 843eef4 (Phase 1 integration).

### Phase 2: FFN as unified MTFP21 custom op — COMPLETE

The entire FFN block replaced with one `ggml_map_custom1` call to `shirley_ffn_compute`. Internally, every value is MTFP21 at 25.4-bit precision. The int8 appears ONLY at the matmul SIMD interface — a wire format, not a precision boundary.

Operations fused: ffn_norm (MTFP21 RMSNorm) → gate/up matmuls (pack→sign_epi8→MTFP21) → ReLU (MTFP21) → square (MTFP21) → element-wise mul (MTFP21) → ffn_sub_norm (MTFP21 RMSNorm) → down matmul (pack→sign_epi8→MTFP21) → residual ADD (MTFP21).

PPL: 17.887 (5-chunk), baseline 17.873 — within noise.

Key insight: int8 quantization between every op lost 1+ PPL. MTFP21 intermediates: zero PPL loss. The matmul wire format is not a precision decision.

Commits: 75e24d8 (scaffold), 937fe7f (gemv fix + batch), b0489e1 (MTFP21 rewrite).

### Phase 3: Integer RoPE

Precompute sin/cos tables as Q15 int16 at model load. Build `shirley_rope_i8()` — per-element: two `mulhrs_epi16` + `add_epi16` + pack to i8.

**This completes:** Q and K preparation in integer.

### Phase 4: Integer attention matmuls

Build `shirley_matmul_i8_i8()` for Q@K^T and attn@V. Standard int8×int8→int32 matmul using AVX2 `_mm256_maddubs_epi16` + `_mm256_madd_epi16`. MTFP21 scale tracking.

**This completes:** Attention score computation and context aggregation in integer.

### Phase 5: Softmax — the EXP prime

MTFP21 exp() is BUILT (commit 6bb5ccb): LUT + linear interpolation, 256 entries, max error 9.5e-6. MTFP21 softmax is BUILT: subtract max → exp → sum → normalize → quantize. 109/109 tests pass.

Remaining: wire into the attention custom op (same pattern as FFN — one function, MTFP21 throughout). CPU fallback first, iGPU dispatch (V_EXP_F32) is an optimization.

**This completes:** The attention mechanism end-to-end without float32.

### Phase 6: Embedding + LM head

Quantize embedding table to int8 + MTFP21 scale at model load. Token lookup returns int8 directly. LM head reuses the quantized table for the final matmul.

Sampling uses the Phase 5 softmax.

**This completes:** End-to-end ternary/MTFP21 inference. Zero float32 in the compute path.

## Data Format Transitions

```
Embedding (i8 + MTFP21 scale)

Per layer:
  i8 → RMSNorm(i8→i8) → matmul(i8×2bit→i32) → rescale(i32→i8)

  For Q,K: i8 → RoPE(i8→i8, Q15 sin/cos) → i8
  Attention: i8 Q × i8 K^T → i32 scores
           → softmax(i32→i8, EXP via iGPU)
           → i8 weights × i8 V → i32 → rescale → i8

  i8 → RMSNorm → i8 → matmul(wo) → i32 → rescale → i8
  i8 + i8 → residual → i8

  i8 → RMSNorm → i8 → matmul(gate,up) → i32 → rescale → i8
  i8 → ReLU → i8 → Square → i16
  i16 × i8 → i32 → rescale → i8
  i8 → RMSNorm → i8 → matmul(down) → i32 → rescale → i8
  i8 + i8 → residual → i8

Output:
  i8 → RMSNorm → i8 → matmul(LM head, i8×i8→i32) → softmax(iGPU) → token
```

Every value is integer. Every scale factor is MTFP21. Every transcendental routes to iGPU. Every transition has an MTFP21 scale that tracks quantization.

## Scale Factor Bookkeeping

Every int8 tensor carries an MTFP21 scale factor:
```
real_value = int8_value × mtfp21_scale
```

| Scale | Source | When computed | Format |
|-------|--------|--------------|--------|
| embd_scale | Embedding quantization | Model load | MTFP21 per-row |
| act_scale | RMSNorm output | Per RMSNorm call | MTFP21 |
| weight_scale | Model weights | Model load | MTFP21 per-layer |
| combined_scale | weight_scale / act_scale | Model load (static) or per-call | MTFP21 → Q15 |
| wo/gate/up/down_scale | Model learned scalars | Model load | Folded into combined_scale |
| norm_scale | RMSNorm rsqrt | Per RMSNorm call | MTFP21 → Q15 via mulhrs |
| rope_sin/cos | Position encoding | Model load | Q15 int16 table |
| attn_scale | Attention score normalization | Per attention call | MTFP21 |

All scalar. All precomputable or computable in MTFP21 with one scalar operation. The per-element work is always AVX2 integer.

## What the Six Primes Do Here

| Prime | Where it appears | Substrate |
|-------|-----------------|-----------|
| ADD | Residual connections, matmul accumulation, bias | CPU (AVX2 add_epi8/16/32) |
| MUL | Ternary matmul (sign_epi8), RoPE, gamma, rescale | CPU (AVX2 sign_epi8, mulhrs_epi16) |
| MAX | ReLU, softmax stabilization (subtract max), routing | CPU (AVX2 max_epi8) |
| CONST | Scale factors, sin/cos tables, gamma weights, thresholds | CPU (immediate/load) |
| EXP | Softmax | **iGPU** (V_EXP_F32, hardware, ~4 cycles) |
| LOG | Not used in this model's forward pass | **iGPU** (available for future ops) |

Four primes on CPU. One prime on iGPU. LOG is available but not needed in this architecture (no log-softmax, no entropy computation in the forward pass).

## Origin

Integration map created April 1, 2026 from Session 002 architecture audit of `build_bitnet_158()` in `llama.cpp:15389-15537`, `llm_build_ffn()` in `llama.cpp:9526-9651`, and `llm_build_kqv()` in `llama.cpp:9754-9882`. Every operation verified against the actual model forward pass code.

Revised April 1, 2026 to remove all float32 from the target pipeline. The Shirley thesis demands integer/MTFP21/iGPU throughout. Float32 is what we're replacing, not what we're accommodating.
