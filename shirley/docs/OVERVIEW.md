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
