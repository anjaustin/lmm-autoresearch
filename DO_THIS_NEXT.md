# DO THIS NEXT

## Current Status (2026-04-02)

The core pipeline runs adaptive-width MTFP with zero float in the matmul path.

```
Embedding (float32 → MTFP21 at layer 0)
→ 30 layers × [
    MTFP16 attention (sign_epi16 QKV+wo, MTFP21 norm/RoPE/softmax/dot)
    MTFP16 FFN (sign_epi16 gate/up/down, MTFP21 norm/ReLU²/mul)
  ]
→ MTFP21 output norm
→ LM head (float32 — output boundary)
→ Sampling
```

All 210 ternary matmuls use `sign_epi16` with int16 mantissas. Zero `mtfp21_to_float` or `mtfp21_from_float` in the matmul path.

### Adaptive-Width MTFP Family

```
MTFP21  int32 mantissa (25.4 bits)  scalar    RMSNorm rsqrt, softmax exp, RoPE, between-matmul ops
MTFP16  int16 mantissa (15.8 bits)  16 lanes  matmul wire via sign_epi16 + block exponent
MTFP8   int8  mantissa              32 lanes  (available, not used — MTFP16 is the production wire)
```

Same base-3 exponent everywhere. The geometric coordinate is invariant across widths. Width transitions are truncation (21→16) or extension (16→21) — the exponent doesn't change.

### What's Done

- **MTFP arithmetic:** 109/109 tests (add, mul, div, rsqrt, exp, softmax, cmp)
- **MTFP16 matmul kernel:** `shirley_mtfp16_matmul.h` — sign_epi16 × 2-bit packed ternary, 6/6 tests, statistical validation (avg error 8e-05)
- **Attention custom op:** attn_norm + QKV (sign_epi16) + RoPE (CONST) + Q@K^T (MTFP21) + softmax (EXP) + attn@V (MTFP21) + sub_norm + wo (sign_epi16) + residual
- **FFN custom op:** ffn_norm + gate/up (sign_epi16) + ReLU² + mul + sub_norm + down (sign_epi16) + residual
- **Output norm:** MTFP21 RMSNorm
- **Generation:** Coherent, factually accurate across diverse prompts
- **Geometric insight:** Numbers are positions. Exponents are coordinates. Documented.

### What Remains: Float32 Remediation

Two float32 operations remain. Both are at the model boundary between discrete tokens and continuous computation.

**1. Embedding lookup (input boundary)**
- `model.tok_embd` is float32 [2560 × 128256]
- Lookup produces float32 vector, immediately converted to MTFP21 at layer 0
- Remediation options:
  - (a) Convert embedding table to MTFP21 at model load. Lookup returns MTFP21 directly.
  - (b) Accept: the lookup is one operation per token, and the float→MTFP21 conversion at layer 0 is already happening. The float is transient.
- Blocker: the 128K × 2560 embedding table is 1.3 GB as float32, stored memory-mapped. Accessing all rows from a custom op segfaults at ~75% (ggml memory management issue). Needs investigation.

**2. LM head matmul (output boundary)**
- `model.tok_embd` (tied weights) × normalized output → logits [128256]
- Currently ggml float matmul because custom ops can't change output shape ([2560] → [128256])
- Remediation options:
  - (a) Shape donor tensor via ggml_map_custom2 (validated — shape works, embedding access blocks)
  - (b) Bypass ggml: allocate output buffer, run MTFP matmul, hand float logits to sampling
  - (c) Accept: logits are the last continuous values before discrete token selection. Float at this boundary is the conversion from MTFP21 geometry to probability space.
- Blocker: same embedding access issue as (1).

**3. KV cache stores float (internal)**
- `mtfp21_to_float` on cache write, `mtfp21_from_float` on cache read
- Remediation: store `mtfp21_t` structs directly in the cache. This makes the geometric interpretation real — each cache entry preserves its position coordinate (exponent).
- No blocker. Straightforward struct change.

**4. RoPE tables store float (internal)**
- Sin/cos precomputed as float, converted to MTFP21 per-element at runtime
- Remediation: precompute as MTFP21 at model load. Eliminates per-element `mtfp21_from_float`.
- No blocker. Straightforward.

### Priority

1. KV cache → MTFP21 native (no blocker, makes geometry real)
2. RoPE tables → MTFP21 (no blocker, eliminates conversion)
3. LM head (needs embedding access fix or bypass)
4. Embedding lookup (same blocker as LM head)

## Lessons

1. **Adaptive-width MTFP.** Same exponent, different mantissa. MTFP21 for precision, MTFP16 for SIMD, MTFP8 available. The exponent is the invariant.
2. **sign_epi16 is the ternary matmul instruction.** Not sign_epi8. The wider mantissa preserves precision that int8 crushed.
3. **Zero float in the matmul path.** Block-align MTFP21 → int16 mantissas → sign_epi16 → int32 → MTFP21. No mtfp21_to_float anywhere.
4. **MTFP21 intermediates, not int8.** Quantizing between ops lost 1+ PPL. MTFP21: zero loss.
5. **Custom ops beat patching.** One function per block. Don't fight the framework — bypass it.
6. **Don't accommodate float32, replace it.** Every retreat to float made the architecture worse.
7. **Numbers are positions.** The exponent is a coordinate. The graph is the geometry.
8. **The LMM finds what you're not seeing.** The third vectorization approach (adaptive width) came from the manifold, not from incremental engineering.
