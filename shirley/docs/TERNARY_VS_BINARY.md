# Computing in Ternary vs Computing in Binary

## The Structural Difference

Binary is two symbols: 0, 1. One bit. Two states per digit.

Ternary is three symbols: -1, 0, +1. One trit. Three states per digit.

The difference isn't "one more symbol." It's structural.

### Sign is native

Binary is unsigned by nature. To represent negative numbers, you bolt on a convention — two's complement, sign bit, offset encoding. The hardware doesn't know what "negative" means. You teach it through interpretation.

Ternary is signed by nature. {-1, 0, +1} has polarity built into the symbol set. Negation is multiplication by -1 — flipping -1 to +1 and vice versa. No convention needed. The representation *is* the meaning.

### Zero is a first-class citizen

In binary, every digit carries signal. 0 and 1 are both "something." A zero bit is as expensive to store and process as a one bit.

In ternary, 0 means "this doesn't matter." It's sparsity as a primitive. A ternary weight vector that's 75% zeros isn't a compressed dense vector — it's a vector that *only cares about 25% of its inputs*. Sparsity isn't an optimization applied after the fact. It's what the representation naturally expresses.

### Multiplication collapses to routing

Multiplication in binary requires a multiplier circuit — dedicated hardware, multiple cycles for wide operands, carry propagation.

Multiplication in ternary {-1, 0, +1} is:
- × (+1): pass through
- × (0): zero
- × (-1): negate

No multiplier. The operation is selection and sign flip. This is what MAX and a conditional negate give you. This is why TriX can eliminate MatMul — ternary multiplication is routing, not arithmetic.

### Information density

log₂(3) ≈ 1.585 bits per trit. A trit carries 58.5% more information than a bit.

| Width | Binary states | Ternary states | Ratio |
|-------|--------------|----------------|-------|
| 1 | 2 | 3 | 1.5× |
| 2 | 4 | 9 | 2.25× |
| 4 | 16 | 81 | 5.06× |
| 8 | 256 | 6,561 | 25.6× |
| 16 | 65,536 | 43,046,721 | 656× |

The ternary representation is denser *and* natively signed *and* natively sparse.

## Mapping the Six Primes

| Prime | Binary implementation | Ternary implementation | Difference |
|-------|---------------------|----------------------|------------|
| ADD | Full adder, carry propagation | Trit addition: 9-entry lookup, bounded range | Simpler |
| MUL | Multiplier circuit (expensive) | Sign flip or zero — **no multiplier** | Dramatically simpler |
| MAX | Comparator + mux | Trit comparison: single lookup | Simpler |
| CONST | Bit pattern | Trit pattern — natively encodes sign | Equivalent |
| EXP | Transcendental (continuous domain) | Same — operates on continuous output | Identical |
| LOG | Transcendental (continuous domain) | Same — operates on continuous output | Identical |

The first four primes are cheaper in ternary. MUL is the biggest win — it goes from a hardware multiplier to a conditional sign flip. EXP and LOG don't change because they operate on continuous values after the ternary domain has done its work.

## The Hardware Proof: `_mm256_sign_epi8`

The x86 AVX2 instruction set contains an instruction that *is* ternary multiplication:

```c
__m256i result = _mm256_sign_epi8(a, b);
```

**Semantics:**
- If `b > 0`: result = `a` (multiply by +1)
- If `b == 0`: result = `0` (multiply by 0)
- If `b < 0`: result = `-a` (multiply by -1)

For ternary values {-1, 0, +1}, this is exact multiplication. Not an approximation. Not a workaround. The instruction does precisely what ternary MUL means.

**Verification:**

| a | b | sign_epi8(a,b) | a × b |
|---|---|----------------|-------|
| +1 | +1 | +1 | +1 |
| +1 | 0 | 0 | 0 |
| +1 | -1 | -1 | -1 |
| -1 | +1 | -1 | -1 |
| -1 | 0 | 0 | 0 |
| -1 | -1 | +1 | +1 |
| 0 | +1 | 0 | 0 |
| 0 | 0 | 0 | 0 |
| 0 | -1 | 0 | 0 |

All nine ternary multiplication cases are correct. The instruction computes 32 ternary products in a single cycle.

**Performance:**
- 32 ternary multiplications per cycle per lane
- 800 ternary products (25 iterations × 32 lanes) in ~25 cycles
- ~8 nanoseconds per full dot product at 3 GHz
- Compared to float32 multiply: same throughput, zero precision loss, integer pipeline

## The Ternary Dot Product

The fundamental operation of neural inference — the dot product — reduces to `sign_epi8` + `add_epi8`:

```c
static inline int32_t ternary_dot(const int8_t *a, const int8_t *b) {
    __m256i acc = _mm256_setzero_si256();
    for (int i = 0; i < PADDED; i += 32) {
        __m256i va = _mm256_load_si256((const __m256i *)(a + i));
        __m256i vb = _mm256_load_si256((const __m256i *)(b + i));
        __m256i prod = _mm256_sign_epi8(va, vb);  // ternary multiply
        acc = _mm256_add_epi8(acc, prod);          // accumulate
    }
    // horizontal sum acc → int32
}
```

No floating point. No multiplier. Two instructions per 32 elements: sign + add. The entire multiply-accumulate of neural inference collapses to select-and-accumulate.

## Why Routing Is Natural in Ternary

The signature matching in TriX — `input @ signature` where both are ternary — is just a ternary dot product:

```
signature = weights.sum(dim=0).sign()   // ternary: {-1, 0, +1}
score     = input @ signature           // ternary dot product via sign_epi8
route     = (score == scores.max())     // MAX prime
```

For each dimension:
- Signature is +1: this dimension matters, same polarity
- Signature is 0: this dimension doesn't matter (sparsity)
- Signature is -1: this dimension matters, opposite polarity

The dot product computes "how aligned is this input with what this shape wants?" — using the same `sign_epi8` instruction that does ternary multiplication. The routing primitive and the computation primitive are the *same operation*.

In binary, routing and computation are different circuits. In ternary, they're the same instruction.

## Implications for Shirley

1. **Four of six primes are single-cycle, 32-wide on CPU.** ADD, MUL, MAX, CONST all execute as native AVX2 integer instructions on the Ryzen. The ternary domain makes the CPU substrate more powerful than the binary domain would.

2. **MUL is free.** The most expensive operation in conventional neural inference (matrix multiplication) collapses to sign flips. This fundamentally changes the compute budget — in a fixed time budget, a ternary system can do dramatically more "multiplications" than a binary float system.

3. **Routing is computation.** The operation that selects which shape processes an input (signature matching via dot product) uses the same instruction as the computation within the shape (ternary dot product). There is no overhead for routing. The routing *is* the computation.

4. **Sparsity is free.** A ternary 0 doesn't just skip a multiply — it produces a zero output from `sign_epi8` naturally. Sparse weights don't need special handling. The instruction does the right thing without branching.

5. **EXP and LOG are the only operations that leave the ternary domain.** When a shape needs a transcendental function, the computation exits integer ternary and enters the continuous domain. This is where the iGPU earns its place — handling the domain crossing that the CPU can't do natively.

## The Complete Substrate Map

```
┌──────────────────────────────────────────────────────────┐
│                  TERNARY DOMAIN                           │
│              {-1, 0, +1} — integer, exact                 │
│                                                           │
│  ADD:   _mm256_add_epi8      (CPU, 1 cycle, 32-wide)     │
│  MUL:   _mm256_sign_epi8     (CPU, 1 cycle, 32-wide)     │
│  MAX:   _mm256_max_epi8      (CPU, 1 cycle, 32-wide)     │
│  CONST: immediate / load     (CPU, trivial)               │
│                                                           │
│  Routing: ternary dot product (same as MUL + ADD)         │
│  Sparsity: native (0 = don't care)                        │
│  Sign: native (-1 = negate)                               │
│                                                           │
├──────────────────────────────────────────────────────────┤
│           ↕ DOMAIN CROSSING (precision boundary)          │
├──────────────────────────────────────────────────────────┤
│                                                           │
│               CONTINUOUS DOMAIN                           │
│           float32 — approximate, transcendental           │
│                                                           │
│  EXP:  V_EXP_F32            (iGPU, ~4 cycles, hardware)  │
│  LOG:  V_LOG_F32            (iGPU, ~4 cycles, hardware)  │
│                                                           │
│  Shared memory: no PCIe transfer needed (APU)             │
│                                                           │
└──────────────────────────────────────────────────────────┘
```

The six primes partition naturally: four in the ternary integer domain (CPU), two in the continuous transcendental domain (iGPU). The domain crossing — where ternary values become continuous for EXP/LOG and return — is the precision boundary that b158's dithering research mapped.
