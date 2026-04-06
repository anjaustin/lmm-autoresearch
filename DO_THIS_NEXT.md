# DO THIS NEXT

## Current Status (2026-04-06)

Native MTFP21 throughout the between-matmul path. No format conversions between operations.

```
Embedding (float32)
→ mtfp21_from_float (model boundary)

Per layer:
  ATTENTION (single-threaded):
    MTFP21 → rmsnorm → block-align → [int16 matmul wire] → MTFP21 (QKV)
    MTFP21: RoPE, cache, Q@K^T, softmax, attn@V (native MTFP21 ops)
    MTFP21 → rmsnorm → block-align → [int16 matmul wire] → MTFP21 (wo)
    MTFP21: residual ADD
    → mtfp21_to_float (ggml boundary)

  FFN (multi-threaded matmuls):
    mtfp21_from_float (ggml boundary)
    MTFP21 → rmsnorm → block-align → [int16 matmul wire] → MTFP21 (gate, up)
    MTFP21: mtfp21_relu → mtfp21_square → mtfp21_elem_mul (NATIVE — 3 calls)
    MTFP21 → rmsnorm → block-align → [int16 matmul wire] → MTFP21 (down)
    MTFP21: residual ADD
    → mtfp21_to_float (ggml boundary)

LM head (float32)
```

The int16 exists ONLY at the matmul SIMD wire (`sign_epi16`). Everything else is native MTFP21. The between-matmul trivials (ReLU, square, multiply) are 3 MTFP21 function calls — no conversion, no parallel arrays, no intermediate formats.

### Speed: 4.57 tok/s

| Milestone | tok/s | Change |
|-----------|-------|--------|
| Scalar MTFP21 baseline | 2.14 | — |
| + SIMD matmul + RMSNorm | 3.82 | +79% |
| + FFN multi-threading | 4.33 | +13% |
| + Native MTFP21 trivials | 4.57 | +5.5% |
| Total improvement | | **+114%** |

### What's Done

- **MTFP arithmetic:** 109/109 tests
- **MTFP16 matmul kernel:** sign_epi16 × 2-bit ternary, batched normalization
- **Chunked MTFP21 dot product:** 8-wide SIMD with per-element exponents
- **SIMD RMSNorm:** chunked sum-of-squares + vectorized scale×gamma
- **Native MTFP21 trivials:** mtfp21_relu, mtfp21_square, mtfp21_elem_mul — no conversion
- **FFN multi-threading:** gate+up and down matmuls across 6 threads
- **All constants precomputed as MTFP21:** gamma, eps, kq_scale, RoPE tables, KV cache
- **Generation:** 3/3 core prompts correct, 7/7 extended prompts verified

### What Remains

**1. Attention multi-threading**
The split-node approach (commit `85a616e`) has a regression — tested correctly during the session it was built but fails on subsequent runs. Likely a build artifact or initialization issue. The single-threaded attention works correctly. This needs independent debugging with a clean-room rebuild.

**2. Model boundary float32**
- Embedding lookup (float32 table → mtfp21_from_float)
- LM head matmul (mtfp21_to_float → float32 logits)
- ggml boundary conversions (60× per token)
Blocked by the ggml embedding memory access issue.

**3. Performance optimization**
- The ternary matmul (`shirley_gemv_mtfp16`) is ~30% of total time and already multi-threaded
- The MTFP21 scalar ops (relu, square, mul, RMSNorm rsqrt, softmax exp) are the next targets
- The ggml boundary conversions are ~10% of time (structural)

### Retracted Claims

The 5.27 tok/s claim from the split-node attention commit (`85a616e`) is **unverified**. The binary that produced it may have been stale. The split-node code produces corrupted output on fresh builds. The reliable speed is 4.57 tok/s.

## Lessons

1. **Native MTFP21 between matmuls.** No format conversion. The kernels exist — use them.
2. **The int16 is a wire format.** It exists for sign_epi16 at the matmul boundary. Nowhere else.
3. **Conversion overhead > compute overhead.** Touching 220 KB to convert formats costs more than 13K scalar MTFP21 multiplies on 166 KB of data.
4. **Test on fresh builds.** Stale binaries produce false positives. Rebuild and retest after every commit.
5. **The LMM finds what incremental debugging doesn't.** The split-node insight came from the manifold. The native trivials insight came from Tripp asking "why are we converting?"
