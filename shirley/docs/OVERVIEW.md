# Shirley

Codename for a research project exploring routing-first computation on heterogeneous silicon using the six primes of frozen computation.

## The Thesis

Computation has six irreducible primitives — ADD, MUL, EXP, LOG, MAX, CONST. Every computable function is a composition of these primes into frozen shapes. The shapes don't change. The data doesn't move to the equations — the equations move to the data. Routing is the only degree of freedom.

When you read the hardware honestly, the six primes are physically distributed across two substrates in commodity silicon (AMD Ryzen with integrated GPU):

- **CPU (x86-64/AVX2):** ADD, MUL, MAX, CONST — native, single-cycle, SIMD-wide
- **iGPU (RDNA):** EXP, LOG — native hardware transcendental units, ~4 cycles

Routing doesn't just select which shape to apply. It selects *which silicon* executes it. Heterogeneous compute isn't an optimization bolted on top — it's what the primes demand when the routing reads the hardware topology.

## The Six Primes

| Prime | Role | CPU | iGPU |
|-------|------|-----|------|
| ADD | Linear combination, accumulation | Native (VADDPS, 1 cycle, 8-wide) | Native |
| MUL | Scaling, gating, interaction | Native (VMULPS, 1 cycle, 8-wide) | Native (FMA) |
| EXP | Nonlinearity, softmax, attention | Software (~10-20 cycles, polynomial) | **Hardware (V_EXP_F32, ~4 cycles)** |
| LOG | Inverse of EXP, dynamic range, information theory | Software (~10-20 cycles, polynomial) | **Hardware (V_LOG_F32, ~4 cycles)** |
| MAX | Selection, ReLU, routing decisions | Native (VMAXPS, 1 cycle, 8-wide) | Native |
| CONST | Bias, thresholds, fixed values | Native (MOV immediate) | Native (constant buffer) |

### Why six, not five

The original TriX architecture used five primes (ADD, MUL, EXP, MAX, CONST). LOG was added because EXP without LOG is a one-way street — the exponential domain isn't closed under the original five. No finite composition of ADD, MUL, EXP, MAX, CONST produces a logarithm.

LOG completes the set:

- **Power functions:** `a^b = exp(b * log(a))` — previously impossible to compose
- **Division:** `1/x = exp(-log(x))` via MUL — previously missing
- **SQRT:** `sqrt(x) = exp(0.5 * log(x))` — falls out as a composed shape, no seventh prime needed
- **Information-theoretic quantities:** Entropy, cross-entropy, KL divergence — all fundamentally logarithmic

EXP and LOG are now a matched pair. The six primes are minimal — remove any one and its function cannot be recovered from the remaining five.

## Frozen Shapes

A frozen shape is a fixed composition of primes. It doesn't learn. It doesn't change. It's a mathematical function compiled from the six primitives.

Shapes are the programs. Primes are the instruction set. Routing is the CPU.

Examples:
- **ReLU:** `MAX(x, CONST(0))` — two primes, CPU-native
- **Softmax numerator:** `EXP(x)` — one prime, routes to iGPU
- **Log-softmax:** `x - LOG(EXP(x).sum())` — four primes, mixed substrate
- **RMS normalization:** `x * EXP(CONST(-0.5) * LOG(MUL(x, x).mean()))` — five primes, mixed substrate
- **Sigmoid:** `EXP(x) / (ADD(EXP(x), CONST(1)))` — three primes, routes to iGPU

Shapes compose into larger shapes. A transformer attention head is a composition of shapes. An FFN is a composition of shapes. The entire model is a tree of frozen compositions with routing at every branch.

## Routing

Routing is the only learned (or evolved) component. Everything else is frozen.

From TriX:
```
signature = tile_weights.sum(dim=0).sign()  # What the shape wants
score     = input @ signature               # How well input matches
route     = (score == scores.max())         # Send to best match
```

Ternary weights are votes. Signatures encode preferences. Routing emerges from alignment. "Don't learn what you can read."

### Hardware-aware routing

In Shirley, routing has two dimensions:

1. **Semantic routing:** Which shape should process this input? (from TriX — weight-derived signatures, Hamming distance matching)
2. **Substrate routing:** Which silicon should execute this shape? (from the prime-to-hardware mapping — shapes containing EXP/LOG route to iGPU, shapes composed only of ADD/MUL/MAX/CONST stay on CPU)

Substrate routing is deterministic — it falls out of the shape's prime composition. A shape that contains no EXP or LOG never touches the iGPU. A shape that does always routes its transcendental components there. No decision needed. The routing reads the shape's structure the same way semantic routing reads the weight structure.

"Don't learn what you can read" applies to hardware topology, not just weight signatures.

## MTFP21: The Number System

Shirley operates on MTFP21 — a 21-trit balanced ternary floating point representation that exceeds IEEE 754 float32 in both precision and range:

- **16-trit mantissa:** 25.4 bits precision (float32 = 24 bits), ~7.6 decimal digits
- **5-trit exponent:** dynamic range 10^±57 (float32 = 10^±38)
- **Sign:** free — implicit in balanced ternary. No sign bit needed.
- **Sparsity:** native — zero trits mean "don't care"
- **Arithmetic:** entirely in AVX2 integer pipeline via `sign_epi8`
- **No NaN, no Inf, no denormals, no negative zero.** Every trit pattern is a valid number.

16 mantissa trits = half an AVX2 register. Two MTFP21 mantissas per 256-bit register. The hardware alignment is exact.

See `MTFP21.md` for the full specification.

## Validation: BitNet b1.58-2B-4T (2026-04-01)

The Shirley thesis was tested on a real 2-billion-parameter language model (Microsoft BitNet b1.58-2B-4T) using the autoresearch protocol. Two sessions of experiments across 6 of 8 planned phases, with architecture audit and kernel development.

**Key results (corrected after Session 002 real-integer validation):**

- `sign_epi8` produces mathematically identical results to the existing `maddubs_epi16` kernel — the ternary multiply instruction works as the native matmul primitive
- **5-trit (~161-level) activation quantization is the real floor for generation quality.** Simulation predicted 4-trit; real integer quantization showed 4-trit passes perplexity but causes repetitive generation. PPL alone is insufficient — distribution shape matters. (Session 002 finding)
- **The model uses ReLU-squared, not SiLU.** The earlier "SiLU is free" finding was testing a code path that doesn't exist in this architecture. ReLU-squared (MAX + MUL) is composed entirely of CPU-native primes. (Session 002 architecture audit)
- RMSNorm output is the **precision bottleneck** — simulation shows 7 trits needed (unvalidated with real integer compute; may need correction like the 4-trit finding)
- Residual connections are exact in integer (int16 range sufficient for 28 layers)
- Attention Q@K^T and attn@V are **float×float, not ternary.** Only the projection matmuls (wq, wk, wv, wo, gate, up, down) are ternary. (Session 002 audit)
- 4 RMSNorms per layer (attn, attn_sub, ffn, ffn_sub), not 2. (Session 002 audit)

**Three-layer compute stack (LEMM-derived architecture):**

```
Layer 1: AVX2 SIMD
         sign_epi16: ternary routing at matmul wire (16 lanes)
         mul_epi32: chunked dot products for attention (8 lanes)
         rmsnorm_simd: sum-of-squares + vectorized scale (8 lanes)

Layer 2: Native MTFP21
         Between-matmul ops: mtfp21_relu, mtfp21_square, mtfp21_elem_mul
         Scalar precision: rsqrt (LUT+NR), exp (LUT), softmax, RoPE
         No format conversion. Data stays MTFP21 between matmuls.

Layer 3: Matmul wire (MTFP16)
         Block-aligned int16 mantissas for sign_epi16
         Exists ONLY at the matmul boundary (rmsnorm → block-align → matmul)
```

**AVX2 kernel results (shirley_kernels.h):**

| Kernel | Time (n=2560) | Float ops | Notes |
|--------|--------------|-----------|-------|
| `shirley_rmsnorm_ternary` | 417 ns | **zero** | End-to-end ternary, MTFP21 rsqrt + Q15 scale |
| `shirley_rmsnorm_f32` | 430 ns | all float | 8x faster than ggml scalar, with gamma |
| ggml scalar RMSNorm | 3,300 ns | all float | Current BitNet baseline |
| MTFP21 scalar | 142,000 ns | zero | Proof of concept, not production |

The end-to-end ternary kernel (`shirley_rmsnorm_ternary`) is **faster than the float path** because `_mm256_mulhrs_epi16` (Q15 multiply-round-shift) is one instruction where float needs three (convert, multiply, convert back). Zero float operations between int8 input and int8 output.

**MTFP21 validation (shirley_mtfp21.h):**

109/109 tests pass. Integer-only arithmetic that exceeds float32 precision:
- Accumulation: MTFP21 wins 57.2% of 1000 head-to-head comparisons vs float32
- rsqrt: 256-entry LUT + 2 Newton-Raphson iterations, max error 8.94e-08, zero float
- exp(x): 256-entry LUT + linear interpolation, max error 9.50e-06, handles full softmax range [-30, 0]
- softmax: full MTFP21 (subtract max → exp → sum → normalize), conservation validated
- cmp: integer comparison without float conversion
- RMSNorm: 100% exact match against float64 reference through full int8→MTFP21→int8 pipeline

**Pipeline integration (2026-04-01):**

- **Phase 1 COMPLETE:** All 7 ternary matmuls per layer write int8 via `shirley_rescale_raw_to_i8`. 5-trit activations validated (PPL 18.888, baseline 18.852).
- **Phases 2-5 COMPLETE:** Attention and FFN blocks each run as single custom ops. Adaptive-width MTFP: MTFP21 for precision operations (RMSNorm, softmax, RoPE, between-matmul ops), MTFP16 for the matmul wire (sign_epi16, 16 SIMD lanes). All 210 ternary matmuls across 30 layers use sign_epi16 with zero float conversion. PPL matches baseline.
- **Phase 6 REMAINING:** Embedding lookup and LM head matmul (model boundaries). KV cache and RoPE tables store float (straightforward fix).

Key findings:
1. int8 quantization between ops lost 1+ PPL. MTFP21 intermediates: zero loss.
2. Adaptive-width MTFP: same base-3 exponent at every width. MTFP16 for SIMD, MTFP21 for precision. The exponent (geometric coordinate) is invariant.
3. sign_epi16 replaces sign_epi8 as the production matmul instruction. Wider mantissa (10 trits, 15.8 bits) preserves precision that int8 crushed.

Full results: `BITNET_TERNARY_PLAN.md`. Kernel code: `../bitnet/shirley_kernels.h`, `../bitnet/shirley_mtfp21.h`, `../bitnet/shirley_ffn.cpp`.

## Research Questions

1. **Shape composition search:** What frozen shapes, composed from six primes, best approximate the functions that neural networks learn? Can an evolutionary search (EntroMorph) discover shapes that match or exceed learned approximations?

2. **Routing evolution:** How does the routing structure evolve when shapes are frozen and routing is the only degree of freedom? Does routing discover computational strategies that monolithic learned models can't?

3. **Substrate-aware scheduling:** What is the optimal dispatch strategy for mixed-substrate shapes? When a shape uses both CPU-native and iGPU-native primes, how should the computation be partitioned?

4. **Scaling:** Do frozen shape compositions scale? Can a tree of frozen shapes, each individually simple, compose into computations that rival learned neural networks at meaningful scale?

5. **The LEMM loop:** Can the Lincoln-Einstein Manifold Method (autoresearch protocol with micro-manifold reasoning and arc archiving) accelerate the discovery of optimal shape compositions and routing strategies?

## Connection to Prior Work

| Project | What Shirley inherits |
|---------|----------------------|
| TriX | Five primes (now six), frozen shapes, emergent routing, weight-derived signatures |
| b158 | Precision boundary handling — what happens when continuous values enter the ternary domain |
| FLYNNCONCEIVABLE | Proof that frozen shapes can compose into arbitrary computation (6502 ALU) |
| fungible-computation | The thesis — neural and classical computation are interchangeable representations |
| trixVII | EntroMorph evolutionary search, Gluttony Penalty, zero-latency predictor result |
| SSTT | Zero-parameter classification — proof that learned parameters aren't always necessary |
| LCVDB | Hamming distance routing in practice, bounded-rank theorem |
| L-Cache kernels | Ternary opcode kernel, hardware-resident computation |
| Delta Observer | Scaffolding hypothesis — transient structure during learning, observational steering |

## Target Hardware

Phase 1: AMD Ryzen with integrated GPU (the machine on the desk)
- CPU: Ryzen (x86-64, AVX2, SSE4.2)
- iGPU: AMD integrated graphics (RDNA or Vega)
- Shared memory architecture — CPU and iGPU share physical RAM, eliminating PCIe transfer overhead for heterogeneous dispatch

This is not a limitation. This is the design constraint that forces the routing to be intelligent about substrate selection. A machine with a discrete H100 can brute-force everything on one substrate. A Ryzen with an iGPU *must* route well to perform well.

## Origin

Conceived March 31, 2026 by A.N. Josserand-Austin and Claude during the development of the lmm-autoresearch protocol. Named Shirley because — surely — you can't be serious.
