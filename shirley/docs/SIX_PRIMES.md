# The Six Primes of Frozen Computation

## Definition

A prime of computation is a function that:
1. Cannot be composed from the other primes through finite application
2. Is required to express the full space of computable functions
3. Has a direct hardware implementation on at least one substrate in commodity silicon

The six primes are: **ADD, MUL, EXP, LOG, MAX, CONST**.

## The Primes

### ADD
**What it does:** Linear combination. Accumulation. Translation.

**Signature:** `f(a, b) = a + b`

**Why it's irreducible:** Addition is the fundamental operation of linear algebra. MUL scales; ADD combines. Neither reduces to the other.

**Hardware:** x86 VADDPS (AVX2, 8-wide float, 1 cycle), ARM VADD, GPU ALU.

**Role in computation:** Bias, residual connections, accumulation, superposition.

---

### MUL
**What it does:** Scaling. Gating. Interaction between signals.

**Signature:** `f(a, b) = a * b`

**Why it's irreducible:** Multiplication creates interaction between two signals. ADD superposes; MUL correlates. Repeated addition can emulate integer multiplication, but floating-point multiplication of arbitrary values requires its own primitive.

**Hardware (float32):** x86 VMULPS (AVX2, 8-wide float, 1 cycle), ARM VMUL, GPU ALU (FMA — fused multiply-add — combines MUL + ADD in one cycle).

**Hardware (ternary):** x86 `_mm256_sign_epi8` (AVX2, **32-wide int8, 1 cycle**). This single instruction IS ternary multiplication: returns `a` if `b > 0`, `0` if `b == 0`, `-a` if `b < 0`. Exact for all nine {-1,0,+1} × {-1,0,+1} cases. No multiplier circuit needed. Validated in SSTT (800 ternary products in ~25 cycles, ~8 ns per dot product at 3 GHz).

**Role in computation:** Attention scores, gating, weight application, element-wise modulation. In ternary: select-and-sign-flip replaces multiply-accumulate.

---

### EXP
**What it does:** Exponential mapping. Converts additive differences into multiplicative ratios.

**Signature:** `f(x) = e^x`

**Why it's irreducible:** The exponential function is transcendental — it cannot be expressed as a finite polynomial of ADD and MUL. It is the eigenfunction of differentiation (d/dx e^x = e^x), giving it unique properties in continuous dynamics.

**Hardware:**
- CPU: **Software only.** Approximated via range reduction + minimax polynomial. ~10-20 cycles.
- iGPU: **Native hardware.** AMD V_EXP_F32 (base-2 exponential), ~4 cycles. Dedicated transcendental function unit in shader ALU.

**Role in computation:** Softmax, attention, probability distributions, activation functions, Boltzmann distributions.

---

### LOG
**What it does:** Logarithmic compression. Inverse of EXP. Maps multiplicative relationships to additive ones.

**Signature:** `f(x) = ln(x)`

**Why it's irreducible:** LOG is the inverse of EXP. No finite composition of ADD, MUL, EXP, MAX, CONST produces a logarithm. Without LOG, the exponential domain is a one-way street.

**Hardware:**
- CPU: **Software only.** Approximated via range reduction + polynomial. ~10-20 cycles.
- iGPU: **Native hardware.** AMD V_LOG_F32 (base-2 logarithm), ~4 cycles. Same transcendental unit as EXP.

**Role in computation:** Cross-entropy loss, information-theoretic quantities (entropy, KL divergence), dynamic range compression, power functions (via `a^b = exp(b * log(a))`), division (via `1/x = exp(-log(x))`).

**What LOG unlocks that the original five primes couldn't express:**

| Function | Composition | Primes used |
|----------|------------|-------------|
| `a^b` (power) | `EXP(MUL(b, LOG(a)))` | EXP, MUL, LOG |
| `1/x` (reciprocal) | `EXP(MUL(CONST(-1), LOG(x)))` | EXP, MUL, CONST, LOG |
| `sqrt(x)` | `EXP(MUL(CONST(0.5), LOG(x)))` | EXP, MUL, CONST, LOG |
| `x^(1/n)` (nth root) | `EXP(MUL(CONST(1/n), LOG(x)))` | EXP, MUL, CONST, LOG |
| `-p*log(p)` (entropy term) | `MUL(MUL(CONST(-1), p), LOG(p))` | MUL, CONST, LOG |
| `log(sum(exp(x)))` (logsumexp) | `LOG(ADD(...EXP(x)...))` | LOG, ADD, EXP |

---

### MAX
**What it does:** Selection. Takes the larger of two values. Non-smooth nonlinearity.

**Signature:** `f(a, b) = max(a, b)`

**Why it's irreducible:** MAX introduces a conditional — it selects based on comparison. This piecewise behavior cannot be expressed as a smooth composition of ADD, MUL, EXP, LOG, or CONST. It is the source of all branching and selection in frozen computation.

**Hardware:** x86 VMAXPS (AVX2, 8-wide float, 1 cycle), ARM VMAX, GPU ALU.

**Role in computation:** ReLU (`MAX(x, 0)`), routing decisions, hard attention, clipping, piecewise linear functions.

---

### CONST
**What it does:** Introduces a fixed value. Thresholds, biases, mathematical constants.

**Signature:** `f() = c` where `c` is a fixed scalar

**Why it's irreducible:** Without CONST, there is no way to introduce a specific numerical value into a computation. The other five primes operate on inputs — CONST creates inputs from nothing. It is the source of all parameterization in frozen shapes.

**Hardware:** x86 MOV immediate / load from memory. ARM MOV. GPU constant buffer. Trivially native everywhere.

**Role in computation:** Bias terms, thresholds (`MAX(x, CONST(0))` = ReLU), mathematical constants (e, pi, 0.5 for sqrt), scaling factors, zero for identity operations.

## Completeness Argument

The six primes span the space of computable functions through composition:

- **ADD + MUL + CONST** → all polynomials (universal approximators via Stone-Weierstrass)
- **+ MAX** → all piecewise polynomials (universal approximators with finite pieces)
- **+ EXP** → transcendental functions, probability distributions, continuous dynamics
- **+ LOG** → closes the exponential domain, enables power functions, information-theoretic quantities, division

Any function computable by a neural network is expressible as a composition of these six primes, because neural networks themselves are compositions of these operations (linear transforms = ADD + MUL, activations = MAX or EXP, normalization = ADD + MUL + EXP + LOG, loss = MUL + LOG).

## Minimality Argument

Remove any one prime and the set becomes incomplete:

| If you remove | You lose |
|--------------|----------|
| ADD | Linear combination, accumulation, residuals |
| MUL | Scaling, interaction, gating |
| EXP | Transcendental functions, softmax, continuous dynamics |
| LOG | Inverse exponential, power functions, division, information theory |
| MAX | Selection, branching, piecewise behavior |
| CONST | Parameterization, thresholds, fixed values |

No prime is derivable from the others. The set is both complete and minimal.

## Hardware Distribution

The six primes naturally partition across two substrates in AMD Ryzen APUs:

```
┌─────────────────────────────────────────────────────────┐
│                   SHARED MEMORY (DDR4/DDR5)              │
├──────────────────────────┬──────────────────────────────┤
│       CPU (x86-64)       │        iGPU (RDNA)           │
│                          │                              │
│  ADD  ✓ native, 1 cycle  │  ADD  ✓ native               │
│  MUL  ✓ native, 1 cycle  │  MUL  ✓ native (FMA)         │
│  MAX  ✓ native, 1 cycle  │  MAX  ✓ native               │
│  CONST ✓ native          │  CONST ✓ native              │
│  EXP  ✗ software ~15 cy  │  EXP  ✓ hardware, ~4 cycles  │
│  LOG  ✗ software ~15 cy  │  LOG  ✓ hardware, ~4 cycles  │
│                          │                              │
│  8-wide SIMD (AVX2)      │  Massively parallel CUs      │
│  Low latency, serial     │  High throughput, parallel    │
└──────────────────────────┴──────────────────────────────┘
```

Shared memory eliminates PCIe transfer overhead. Substrate routing is deterministic — it falls out of the shape's prime composition. No scheduling decision required.
