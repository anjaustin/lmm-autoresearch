# DO THIS NEXT

## Current Status (2026-04-01, late session)

The per-layer loop is two Shirley custom ops. Zero ggml float ops inside.

```
inpL → [Shirley attention: norm+QKV+RoPE+attn+sub_norm+wo+residual]
     → [Shirley FFN: norm+gate/up+relu²+mul+sub_norm+down+residual]
     → next layer
```

Both ops run in MTFP21 internally. Int8 only at the matmul SIMD wire.

### What's Done

**Phases 1-5: COMPLETE**
- MTFP21 arithmetic: add, mul, div, rsqrt, exp, softmax, cmp (109/109 tests)
- Phase 1: post-matmul int8 bridge on all 7 ternary matmuls
- Phase 2: FFN as single custom op, fully MTFP21 (PPL matches baseline)
- Phase 3: integer RoPE via precomputed CONST sin/cos tables
- Phase 4: integer attention (Q@K^T + attn@V as MTFP21 dot products)
- Phase 5: MTFP21 softmax (EXP prime) in the attention path
- attn_norm, attn_sub_norm, wo matmul, wo_scale all folded into attention op
- Attention residual ADD folded into attention op

**Generation quality: VERIFIED**
- Coherent across diverse prompts (factual, scientific, creative, technical)
- Factual accuracy maintained (Paris, Heisenberg, etc.)
- No loops, no garbage, good lexical diversity

### What's Remaining (Phase 6: Model Boundaries)

The float32 remaining is at the edges of the model, not inside the layers:

1. **Embedding lookup** — `model.tok_embd` is float32. First operation on input tokens.
   - Fix: quantize embedding table to int8 + MTFP21 scale at model load
   - Or: convert float32 embedding output to MTFP21 at layer 0 input

2. **Output norm** — `llm_build_norm` after the last layer. Float32 RMSNorm.
   - Fix: replace with MTFP21 RMSNorm (same kernel as the layer norms)

3. **LM head matmul** — `model.output` or `model.tok_embd` (tied weights), float32.
   - Fix: MTFP21 → int8, matmul against quantized embedding table
   - Or: convert MTFP21 output to float for this one matmul (it's a model boundary)

4. **Matmul boundary float** — `mtfp21_to_float` and `mtfp21_from_float` at pack/unpack
   - This is transient format conversion, not persistent float storage
   - Fix: native MTFP21 pack/unpack without float intermediary

5. **KV cache stores float** — `mtfp21_to_float` on cache write, `mtfp21_from_float` on read
   - Fix: store `mtfp21_t` structs directly (makes the geometric interpretation real)

### Red-Team Findings

See `shirley/docs/RED_TEAM_20260401.md` for full assessment. Key points:
- The inter-op MTFP21 path is genuine — zero PPL loss vs baseline
- The matmul boundary uses float as format conversion (transient, not persistent)
- The KV cache is float, not MTFP21-native (geometric interpretation not yet implemented)
- Speed is ~1.5 tok/s (scalar MTFP21 ops) — correctness first, performance later

### Architectural Insight: GEOMETRIC_MTFP21.md

Numbers are positions, not quantities. The MTFP21 exponent is a coordinate in base-3 geometric space. Multiply is translation. Add is co-location. The computational graph is intrinsic to the values. The KV cache is a set of landmarks. Attention is navigation. See `shirley/docs/GEOMETRIC_MTFP21.md`.

## Lessons

1. **MTFP21 intermediates, not int8 intermediates.** Quantizing to int8 between ops lost 1+ PPL. MTFP21: zero loss.
2. **The int8 is a wire format.** The matmul SIMD bus is int8 × ternary. That's hardware, not precision.
3. **Custom ops beat patching.** One function per block > modifying 8 ggml dispatch points.
4. **Don't accommodate float32, replace it.** Every time we defaulted to "keep float here," the architecture got worse.
5. **gemv: nc is output rows, nr is unused.** Getting the parameter order wrong produces garbage.
6. **ggml tensor custom fields don't survive graph compilation.** Use op_params or userdata.
7. **Numbers are positions.** The exponent is a coordinate. The graph is the geometry.
