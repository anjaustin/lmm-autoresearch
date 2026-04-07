# DO THIS NEXT

## Current Status (2026-04-06)

Kernel-based pipeline: sign_epi8 matmuls (32 lanes), AVX2 integer trivials, float attention body. Multi-threaded QKV+wo matmuls (all threads), attention body + FFN trivials on thread 0.

```
Embedding (float32, ggml)

Per layer:
  ATTENTION (multi-threaded QKV+wo):
    float → shirley_rmsnorm_quantize → int8                [thread 0]
    int8 × 2bit → sign_epi8 QKV matmuls → float           [ALL threads]
    float: RoPE (precomputed MTFP21→float tables)          [thread 0]
    float: KV cache store                                   [thread 0]
    float: Q@K^T dot product                                [thread 0]
    float: softmax (expf)                                   [thread 0]
    float: attn@V weighted sum                              [thread 0]
    float → shirley_rmsnorm_quantize → int8                [thread 0]
    int8 × 2bit → sign_epi8 wo matmul → float             [ALL threads]
    float: residual ADD                                     [thread 0]

  FFN (multi-threaded gate+up, down):
    float → shirley_rmsnorm_quantize → int8                [thread 0]
    int8 × 2bit → sign_epi8 gate+up matmuls → int32       [ALL threads]
    int32 → rescale → int8                                  [thread 0]
    int8: ReLU (max_epi8), square (mullo_epi16)            [thread 0]
    int16 × int8 → int32 → shift → int16 → requantize     [thread 0]
    int8 → shirley_rmsnorm_ternary → int8 (417ns, Q14)    [thread 0]
    int8 × 2bit → sign_epi8 down matmul → float           [ALL threads]
    float: residual ADD                                     [thread 0]

LM head: MTFP10 int16 GEMV (madd_epi16, multi-threaded, 128K × 2560)
Output norm: float RMSNorm (single-threaded)
```

### Speed: ~22-23 tok/s (4 threads) — matches/exceeds baseline

| Milestone | tok/s | Change |
|-----------|-------|--------|
| Scalar MTFP21 baseline | 2.14 | -- |
| + SIMD matmul + RMSNorm | 3.82 | +79% |
| + FFN multi-threading | 4.33 | +13% |
| + Native MTFP21 trivials | 4.57 | +5.5% |
| + Kernel integration (sign_epi8) | 5.13 | +12% |
| + Multi-threaded attention QKV+wo | 18.89 | **+268%** |
| + MTFP10 LM head (madd_epi16) | 23.02 | **+22%** |
| Baseline BitNet (same model, stock) | ~22 | **matched** |

### Time Budget Per Token (~43ms at 23 tok/s)

| Component | ms | % | Notes |
|-----------|-----|---|-------|
| LM head (MTFP10 int16 GEMV) | ~8 | 19% | 128K x 2560, madd_epi16, multi-threaded |
| FFN (30 layers) | ~22 | 51% | gate+up 51%, trivials 22%, down 22%, sub_norm 3% |
| Attention (30 layers) | ~6 | 14% | QKV 53%, wo 33%, body 4% (grows with seq_len) |
| ggml overhead | ~7 | 16% | dispatch, thread pool, embedding |

### What's Done

- **MTFP arithmetic:** 109/109 tests
- **Kernel-based pipeline:** sign_epi8 matmuls (32 lanes) for all 7 ternary matmuls per layer
- **Multi-threaded attention:** QKV+wo matmuls partitioned across all threads
- **Multi-threaded FFN:** gate+up and down matmuls partitioned across all threads
- **AVX2 trivials:** rescale_i32_to_i8, relu_i8 (max_epi8), square_i8_to_i16, requantize_i16_to_i8
- **shirley_rmsnorm_quantize:** fused float->int8 norm+gamma+quantize for matmul input
- **shirley_rmsnorm_ternary:** int8->int8 norm (417ns, Q14 gamma, zero float) for FFN sub_norm
- **MTFP10 LM head:** Embedding table → block-aligned int16 at load. GEMV via madd_epi16, multi-threaded. ~10ms saved vs float.
- **Adaptive barrier (shirley_barrier.h):** _mm_pause() spin-waits, monotonic phase support
- **Float attention body:** RoPE, Q@K^T, softmax, attn@V in float (hardware FPU)
- **Precomputed MTFP21 tables:** RoPE sin/cos, gamma, eps, kq_scale
- **KV cache:** float (reinterpret-cast of MTFP21 int32 arrays)
- **Profile instrumentation:** per-phase timing with SP_START/SP_LAP/SP_TOKEN

### What Remains

**1. Head-parallel attention body (REVERTED)**
Threading Q@K^T + softmax + attn@V across heads hangs at seq_len ~22. Root cause undiagnosed -- not a data race in the output (head slices don't overlap), not the barrier pattern (same pattern works for matmuls), not the AVX trivials (scalar path also hangs). Needs debugging on a quiet system with debug prints. Currently reverted to thread 0.

**2. FFN trivials threading**
The relu->square->multiply->requantize chain is 22% of FFN time, entirely single-threaded. Threading adds an extra barrier that costs more than it saves at n_ff=6912. The rescale ops need a global max (parallel reduction), which requires yet another barrier. Not profitable at this dimension size.

**3. Barrier optimization**
300 spin-wait barriers per token. Each barrier causes L1 cache line bouncing across cores. Futex-based barriers eliminate the spin but have a phase-cycling race (thread misses a 0->1 transition). sched_yield() causes 1-10ms reschedule delays on loaded systems. Current solution: _mm_pause() spin-wait (reduces power and pipeline stalls, doesn't eliminate cache contention).

**4. LM head dominates at 36% of per-token time**
128,256 x 2,560 float GEMV against f16 embedding table. Memory-bandwidth-bound (~0.61 GB read per token). Both Shirley and baseline pay this equally. Not actionable without model architecture changes.

**5. Model boundary float32**
- Embedding lookup (float32 table)
- LM head (float32 matmul)
- Output norm (MTFP21 scalar, single-threaded)
- ggml boundary conversions (60x per token)

**6. KV cache format**
Stores float in reinterpreted int32 arrays. The exp arrays are allocated but unused (wasted memory). Should either commit to float storage or convert to native MTFP21.

### Retracted Claims

- The 5.27 tok/s claim from split-node attention (85a616e) was RETRACTED (stale binary).
- Head-parallel attention body was reverted due to unexplained hang at seq_len ~22.

### Known Bugs

- **Generation quality:** Model produces incoherent text ("atural Rab titredaily" for "2+2="). Confirmed pre-existing — same garbage with float LM head and MTFP10 LM head. The layer compute (attention + FFN kernels) is producing values that compound incorrectly over 30 layers. Likely a precision or scaling issue in the kernel integration. **This is P0 for the next session.**
- **Phase overflow:** Monotonic phase counter overflows int32 after ~10M tokens. Not a concern for interactive use.
- **Head-parallel hang:** Attention body partitioned across heads hangs at seq_len ~22 with 4 threads. Root cause unknown.

## Lessons

1. **sign_epi8 at 32 lanes is the production matmul.** Not sign_epi16. The 32-lane kernel is what ggml_gemv_i2_i8_s uses.
2. **Threading the matmuls is the big win.** 5->19 tok/s from partitioning QKV+wo+gate+up+down across threads.
3. **Don't thread small arrays.** FFN trivials at n_ff=6912 are too small for profitable threading -- barrier overhead > compute savings.
4. **_mm_pause() in spin-waits is mandatory.** Without it, spinning threads pollute L1 cache and waste power. With it, ~10x power reduction per spin iteration.
5. **sched_yield() is poison on loaded systems.** 1-10ms reschedule delays compound across 300+ barriers per token.
6. **The LM head is 36% of time and untouchable.** Memory-bandwidth-bound. Both Shirley and baseline pay it.
7. **mullo_epi16 overflows silently.** sq (max 6400) x up (max 80) = 512000, doesn't fit in int16. Must widen to int32 before multiply.
8. **Test at 32+ tokens.** Bugs that don't manifest at 8 tokens can hang at 12.
