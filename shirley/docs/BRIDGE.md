# The Bridge: int32 → int8 Post-Matmul Rescale

## The Problem

The ternary matmul (sign_epi8) produces int32 dot products. The current pipeline converts these to float32 via `(raw_int32 / act_scale) * weight_scale` and writes the result to a float32 ggml tensor. Every downstream operation — RMSNorm, ReLU, residual ADD — receives float32 and operates in float32. The data goes back to int8 only when `quantize_row_i8_s` is called before the next matmul.

This float32 intermediate is the ocean between the ternary islands. It costs 4x the memory bandwidth (float32 = 4 bytes vs int8 = 1 byte per element) and prevents the integer kernels from operating directly on matmul output.

## The Solution: Path B

Extend the `ggml_tensor` struct with an optional int8 side-buffer. When the bridge is active, the matmul writes int8 data to the side-buffer instead of (or in addition to) float32 to the main data pointer. Downstream Shirley kernels read from the side-buffer. Operations that don't know about Shirley (RoPE, attention, softmax) read from the float32 data as usual.

This is non-invasive: existing code paths continue to work unchanged. The int8 path is additive.

## Architecture

```
                    ┌──────────────────────────┐
                    │     ggml_tensor           │
                    │                           │
                    │  data: float32*  ────────────→ RoPE, attention, softmax
                    │                           │    (unchanged, reads float)
                    │  shirley_i8: int8*  ─────────→ Shirley RMSNorm, ReLU,
                    │  shirley_scale: float     │    Square, MUL, ADD
                    │  shirley_bridge: bool     │    (reads int8 side-buffer)
                    │                           │
                    └──────────────────────────┘
```

## What Changes

### 1. ggml.h: Extend ggml_tensor

Add three fields to the ggml_tensor struct:

```c
struct ggml_tensor {
    // ... existing fields ...

    // Shirley ternary pipeline: optional int8 side-buffer
    void  * shirley_i8;      // int8 data (NULL if not bridged)
    float   shirley_scale;   // dequant scale: real_value = int8 * shirley_scale
    bool    shirley_bridge;  // true if this tensor should output int8 via bridge
};
```

When `shirley_i8` is non-NULL, the tensor carries int8 data alongside (or instead of) the float32 data. The `shirley_scale` enables reconstruction: `float_value = int8_value * shirley_scale`.

When `shirley_bridge` is true on a matmul destination tensor, the matmul compute function activates the bridge: writing int8 output to `shirley_i8` and setting `shirley_scale`.

### 2. ggml.c: Matmul post-processing (the bridge itself)

In `ggml_compute_forward_mul_mat`, the I2_S post-processing currently does:

```c
// Current (line 13385):
tmp[row] = (tmp[row]) / (act_scales[col]) * (*scale);
// Then: memcpy to float32 dst->data
```

With the bridge:

```c
// Bridge active:
if (dst->shirley_bridge) {
    // Compute combined scale
    float combined_scale = (*scale) / act_scales[col];
    
    // Write int8 to side-buffer via shirley_rescale_i32_scaled_to_i8
    // The tmp[] values are raw int32 dot products (stored as float, but integer-valued)
    int8_t * i8_dst = (int8_t *)dst->shirley_i8 + col_offset;
    float dequant_scale = shirley_rescale_i32_scaled_to_i8(
        i8_dst, tmp_i32, n_rows, out_range, combined_scale);
    dst->shirley_scale = dequant_scale;
    
    // ALSO write float32 for ops that need it (RoPE, attention)
    // This is temporary — once all downstream ops read int8, we skip this
    for (int row = 0; row < n_rows; row++) {
        float_dst[row] = (float)i8_dst[row] * dequant_scale;
    }
}
```

During the transition period, both float32 and int8 data are written. Once all downstream consumers use the int8 path, the float32 write can be removed.

### 3. ggml.c: Downstream dispatch (5 operations)

Each downstream op checks for the int8 side-buffer:

**RMSNorm:**
```c
// In ggml_compute_forward_rms_norm_f32:
if (src0->shirley_i8) {
    // Use shirley_rmsnorm_ternary(dst_i8, src_i8, n, out_range)
    // Set dst->shirley_i8, dst->shirley_scale
} else {
    // Existing float32 path (or Shirley AVX2 float path)
}
```

**ReLU, Square, element-wise MUL, residual ADD:** Same pattern. Check `shirley_i8`, dispatch to int8 kernel if present, fall through to float32 otherwise.

### 4. llama.cpp: Tag which matmuls get the bridge

In `build_bitnet_158()`, after creating each matmul tensor, set the bridge flag:

```c
// QKV projections: NO bridge (output feeds RoPE, needs float)
struct ggml_tensor * Qcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wq, cur);
// Qcur->shirley_bridge = false;  // default

// wo projection: YES bridge (output feeds residual ADD, can be int8)
cur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wo, cur);
cur->shirley_bridge = true;

// gate, up projections: YES bridge (output feeds ReLU² × MUL)
// (set inside llm_build_ffn or after the call)

// down projection: YES bridge (output feeds residual ADD)
cur = llm_build_lora_mm(lctx, ctx0, model.layers[il].ffn_down, cur);
cur->shirley_bridge = true;
```

### 5. Memory allocation for side-buffers

The int8 side-buffer needs memory. Options:

**Option A: Allocate per tensor.** Each bridged tensor gets `malloc(ne0 * ne1 * sizeof(int8_t))` at graph build time. Simple, but adds allocation overhead.

**Option B: Shared workspace.** Allocate one int8 buffer large enough for the largest bridged tensor. All bridged tensors share it (they're computed sequentially, not in parallel). This mirrors how ggml uses `params->wdata` for activation quantization.

**Option C: Pack into existing tensor.** The float32 tensor has `ne0 * ne1 * 4` bytes. The int8 data needs `ne0 * ne1 * 1` bytes. Write int8 into the first 25% of the float32 buffer, float32 into the rest. Fragile but zero-allocation.

**Recommended: Option B.** One shared workspace, allocated once at context creation. The `shirley_i8` pointer in each tensor points into this workspace. Size: `max(ne0 * ne1)` across all bridged tensors. For BitNet-2B: max is 2560 × batch_size bytes.

## Which Matmuls Get Bridged

| Matmul | Bridge? | Why |
|--------|---------|-----|
| wq × attn_norm | NO | Output feeds RoPE (needs float for sin/cos) |
| wk × attn_norm | NO | Output feeds RoPE |
| wv × attn_norm | NO | Output feeds KV cache (float) |
| wo × attn_sub_norm | YES | Output feeds residual ADD (can be int8) |
| ffn_gate × ffn_norm | YES | Output feeds ReLU² (can be int8) |
| ffn_up × ffn_norm | YES | Output feeds element-wise MUL (can be int8) |
| ffn_down × ffn_sub_norm | YES | Output feeds residual ADD (can be int8) |

4 of 7 ternary matmuls per layer get the bridge. The 3 QKV projections stay float because their output enters the float attention mechanism.

## Data Flow With Bridge Active

```
Per layer (with bridge):

  inpL (int8, from previous layer or embedding quantize)
    │
    ├─→ RMSNorm_ternary(attn_norm) → int8 activations
    │     │
    │     ├─→ matmul(wq) → float32 → RoPE → float32 Q
    │     ├─→ matmul(wk) → float32 → RoPE → float32 K  } attention
    │     └─→ matmul(wv) → float32 → float32 V         } (all float)
    │                                    │
    │                                    ▼
    │                              attention(Q,K,V) → float32
    │                                    │
    │                              RMSNorm_f32(attn_sub) → float32
    │                                    │
    │                              matmul(wo) ──BRIDGE──→ int8
    │                                    │
    │     ◄──── residual_add_i8 ◄────────┘
    │
    ├─→ RMSNorm_ternary(ffn_norm) → int8 activations
    │     │
    │     ├─→ matmul(gate) ──BRIDGE──→ int8 → relu_i8 → square_i8→i16
    │     │                                                    │
    │     └─→ matmul(up) ──BRIDGE──→ int8 ────────────────────┤
    │                                                          │
    │                                   mul_i16_i8→i8 ◄────────┘
    │                                        │
    │                                   RMSNorm_ternary(ffn_sub) → int8
    │                                        │
    │                                   matmul(down) ──BRIDGE──→ int8
    │                                        │
    │     ◄──── residual_add_i8 ◄────────────┘
    │
    ▼
  inpL (int8, feeds next layer)
```

The float32 path (QKV → RoPE → attention) is an island. Everything else is int8.

## Implementation Order

1. **ggml.h:** Add 3 fields to ggml_tensor struct
2. **ggml.c:** Initialize new fields to NULL/0/false in tensor creation
3. **llama.cpp:** Set shirley_bridge=true on wo, gate, up, down tensors
4. **ggml.c:** Matmul bridge — write int8 to side-buffer when flag is set
5. **ggml.c:** Downstream dispatch — RMSNorm int8 path
6. **Test:** PPL + generation quality with bridge active on ONE matmul (wo)
7. **Extend:** Add ReLU, Square, MUL, ADD int8 dispatch
8. **Test:** Full FFN block in int8
9. **Test:** Full layer in int8 (except QKV→attention)

Each step gets an inference test before proceeding to the next.

## Verification

After each integration step:
1. Run `builds/test.sh shirley "prompt"` — check generation quality
2. Run perplexity — check PPL is within CI of baseline
3. A/B compare with baseline on same seed — check output is coherent
4. Log everything to `builds/logs/`

## Origin

Bridge design created April 1, 2026 from the integration map analysis. Path B (custom compute function with tensor extension) chosen over Path A (shadow pipeline) per Tripp's direction: "I don't want to do double the work just because one path is easier."
