# Remediation Plan: Native MTFP Throughout

## The Problem

The pipeline converts between formats unnecessarily. The matmul outputs MTFP21. The between-matmul operations convert to int16 parallel arrays for SIMD, then convert back. The conversions are now the bottleneck — 27.6% of FFN time is trivials + sub_norm, dominated by format conversion, not compute.

The MTFP kernels exist. They were built, tested (109/109), and validated. They operate on MTFP21 directly. They're not being used in the hot path.

## The Fix

One format between matmuls. MTFP21 in, MTFP21 operations, MTFP21 out. The int16 block-alignment happens once — at the matmul wire, right before `sign_epi16`. Nowhere else.

## Changes

### 1. `shirley_ffn.cpp` — Remove the MTFP16 conversion in trivials

**Current (lines 229-253):**
```
gate_m (MTFP21) → to_mtfp16 → int16 arrays → relu_simd → square_simd → elem_mul → MTFP21
```

**New:**
```
gate_m (MTFP21) → mtfp21_relu → mtfp21_square → mtfp21_elem_mul → MTFP21
```

Three function calls. No conversion. The functions are in `shirley_mtfp21.h` (already included via `shirley_kernels.h`), already validated.

Remove: `to_mtfp16` loop, `mtfp16_relu_simd`, `mtfp16_square_simd`, `mtfp_elem_mul_32x16` calls.
Replace with: `mtfp21_relu(gate_m, gate_m, n_ff)`, `mtfp21_square(gate_m, gate_m, n_ff)`, `mtfp21_elem_mul(ffn_out, gate_m, up_m, n_ff)`.

These functions exist at lines 104-126 of `shirley_ffn.cpp`.

### 2. `shirley_ffn.cpp` — Simplify the RMSNorm-to-matmul transition

**Current:** `mtfp21_rmsnorm_to_mtfp16` does RMSNorm in MTFP21, then block-aligns to int16 arrays.

**Keep this function as-is.** It's the matmul wire transition — the ONE place where int16 is needed. The RMSNorm computes in MTFP21 (correct), then outputs int16 for the matmul (necessary). This is not a conversion to remove.

### 3. `shirley_ffn.cpp` — Remove VLAs for MTFP16 parallel arrays in trivials

**Remove:** `gate_mant16[n_ff]`, `up_mant16[n_ff]`, `gate_exp16[n_ff]`, `up_exp16[n_ff]`, `sq_mant32[n_ff]`, `sq_exp8[n_ff]` — six VLAs totaling ~60 KB.

**Keep:** `ffn_out[n_ff]` as MTFP21 (the output of the trivials, fed to sub_norm).

### 4. `shirley_attn.cpp` — Same pattern for the attention sub_norm path

The attention body already operates in MTFP21 (Q@K^T, softmax, attn@V). The sub_norm output feeds the wo matmul via block-alignment. No changes needed here — the attention doesn't have the MTFP16 trivials conversion.

### 5. `shirley_mtfp16_ops.h` — No changes needed

The SIMD ops (`mtfp16_relu_simd`, `mtfp16_square_simd`, `mtfp_elem_mul_32x16`) stay in the file for future use (AVX2-vectorized MTFP16 if we need them later). They're just not called from the FFN hot path.

### 6. `shirley_mtfp16_matmul.h` — No changes needed

The matmul kernel stays as-is. It takes block-aligned int16 mantissas and produces MTFP21 output. The int16 input comes from the RMSNorm block-alignment (step 2). The MTFP21 output feeds directly into the trivials (step 1).

## Data Flow After Remediation

```
Embedding (float32)
→ mtfp21_from_float (one conversion, at the model boundary)

Per layer:
  ATTENTION:
    MTFP21 → rmsnorm_to_mtfp16 → [int16 matmul wire] → MTFP21 (QKV)
    MTFP21: RoPE, cache store, Q@K^T, softmax, attn@V (all MTFP21 ops)
    MTFP21 → rmsnorm_to_mtfp16 → [int16 matmul wire] → MTFP21 (wo)
    MTFP21: residual ADD
    → mtfp21_to_float (ggml boundary)

  FFN:
    mtfp21_from_float → MTFP21 (ggml boundary)
    MTFP21 → rmsnorm_to_mtfp16 → [int16 matmul wire] → MTFP21 (gate, up)
    MTFP21: relu, square, elem_mul (THREE MTFP21 FUNCTION CALLS)
    MTFP21 → rmsnorm_to_mtfp16 → [int16 matmul wire] → MTFP21 (down)
    MTFP21: residual ADD
    → mtfp21_to_float (ggml boundary)

LM head (float32)
```

The int16 appears ONLY inside the `rmsnorm_to_mtfp16 → matmul → MTFP21` transition. Everywhere else is native MTFP21.

## Expected Impact

- **Eliminate 60 KB of VLAs** from the FFN trivials (six arrays gone)
- **Eliminate the `to_mtfp16` conversion loop** (6912 × 2 elements × trit-shift per element)
- **Working set drops from ~220 KB to ~110 KB** (gate_m + up_m + ffn_out as MTFP21, no parallel int16 copies)
- **One pass through the data** instead of three (convert → operate → convert)
- **Simpler code** — the trivials section becomes 3 lines instead of 20

## What NOT to Change

- The matmul kernel (`shirley_gemv_mtfp16`) — int16 is the right format for the SIMD wire
- The block-alignment in `mtfp21_rmsnorm_to_mtfp16` — this is the matmul wire transition
- The chunked dot product (`mtfp21_dot_chunked`) — this operates on MTFP21 natively
- The split-node attention threading — this is correct and working
- The FFN multi-threading — the barriers and shared workspace stay as-is

## Files Modified

1. `shirley_ffn.cpp` — trivials section only (~20 lines replaced with ~5)

That's it. One file. One section. The kernels already exist.
