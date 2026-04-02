# DO THIS NEXT

## Current Status (2026-04-01)

The FFN block runs end-to-end in MTFP21. The attention block is next.

### What's Done

**MTFP21 Arithmetic — COMPLETE**
- exp(), softmax, cmp built and validated (109/109 tests, commit 6bb5ccb)
- rsqrt via LUT + Newton-Raphson (102/102 tests)
- Full arithmetic: add, mul, div, neg, abs, from_float, to_float, int8 conversion

**Phase 1: Post-matmul int8 bridge — COMPLETE**
- All 7 ternary matmuls per layer write int8 via `shirley_rescale_raw_to_i8` (commit 843eef4)
- 5-trit activations validated: PPL 18.888, baseline 18.852 (commit bc23f57)
- All 4 code paths covered: 16-row vec_dot, 1-row fallback, gemm, gemv
- The int8 is a wire format for the sign_epi8 SIMD bus, not a precision boundary

**Phase 2: FFN as unified MTFP21 custom op — COMPLETE**
- `shirley_ffn.cpp` replaces 8 ggml ops with one function (commit b0489e1)
- Internally: every intermediate is MTFP21 at 25.4-bit precision
- Matmul interface: MTFP21 → `mtfp21_pack_for_matmul` (int8) → sign_epi8 → MTFP21
- ReLU, square, element-wise multiply all in MTFP21 — no requantization
- PPL: 17.887 (5-chunk), baseline 17.873 — within noise
- Generation: coherent, no loops, good diversity

**Integration Map — REWRITTEN**
- Zero float32 in target pipeline
- Six phases defined, first two complete
- Every operation mapped to integer/MTFP21/iGPU

### What's Next

**Phase 3: Integer RoPE**
- Precompute sin/cos as Q15 int16 tables at model load
- Build `shirley_rope_i8()` — two mulhrs + add + pack per element
- This keeps Q and K in integer after the QKV matmuls

**Phase 4: Integer attention matmuls**
- Build `shirley_matmul_i8_i8()` for Q@K^T and attn@V
- Standard int8×int8→int32 via AVX2 `_mm256_maddubs_epi16`
- MTFP21 scale tracking

**Phase 5: MTFP21 softmax in the attention path**
- `mtfp21_exp()` is built (LUT + linear interpolation, commit 6bb5ccb)
- Wire into attention: subtract max → exp → sum → normalize
- CPU fallback first, iGPU dispatch (V_EXP_F32) later

**Phase 6: Embedding + LM head**
- Quantize embedding table to int8 + MTFP21 scale at model load
- LM head reuses quantized table
- Sampling via MTFP21 softmax

**The end state:** Every value in the forward pass is MTFP21. The int8 lanes are wire format for SIMD. EXP routes to iGPU. Six primes, two substrates.

## File Map

| File | What's there | Status |
|------|-------------|--------|
| `shirley_ffn.cpp` | MTFP21 FFN custom op — the whole block | **ACTIVE** |
| `shirley_ffn.h` | FFN interface declarations | **ACTIVE** |
| `shirley_kernels.h` | AVX2 integer kernels (rmsnorm, relu, sqr, etc.) | BUILT |
| `shirley_mtfp21.h` | MTFP21 arithmetic + exp + softmax + cmp | BUILT |
| `ggml.c` | Phase 1 bridge + rescale function | INTEGRATED |
| `llama.cpp` | FFN custom op wired into build_bitnet_158 | INTEGRATED |
| `ggml/src/CMakeLists.txt` | shirley_ffn.cpp added to build | INTEGRATED |

## Lessons

1. **MTFP21 intermediates, not int8 intermediates.** Quantizing to int8 between every op lost 1+ PPL. Keeping MTFP21 between matmuls: zero PPL loss.
2. **The int8 is a wire format.** The matmul SIMD bus is int8 × ternary. That's a hardware interface, not a precision decision. The exponent rides alongside in MTFP21 bookkeeping.
3. **Custom ops beat patching.** Replacing 8 ggml ops with one Shirley function is cleaner than bolting int8 onto each op individually.
4. **gemv parameter order matters.** `nr` is unused, `nc` is the output dimension. Getting this wrong produces garbage.
5. **ggml tensor fields don't survive graph compilation.** Use `op_params` or `userdata`, not custom struct fields.
