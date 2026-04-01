# MTFP21 Kernel Analysis: How We Lost the Plot

## The Ternary AVX Kernel

The bench_dot kernel does ternary multiply-accumulate in 3 instructions:

```c
__m256i prod = _mm256_sign_epi8(va, vb);   // ternary multiply: 32 elements, 1 cycle
acc = _mm256_add_epi8(acc, prod);            // accumulate: 32 elements, 1 cycle
```

32 elements per cycle. No loops. No branches. No normalization. No widening until
the final horizontal sum. This is what sign_epi8 was designed for — the hardware
does ternary multiplication natively.

Benchmarked: 6.87x faster than float32 mulps at 1024 elements.

## The MTFP21 Kernel

For the same conceptual operation (multiply two values), MTFP21 does:

```c
int64_t prod = (int64_t)a.mantissa * (int64_t)b.mantissa;   // 1. widen to int64
// 2. check if |prod| > MANT_MAX (branch)
// 3. iterative while loop: divide by 3 up to 15 times to find shift amount
// 4. one-shot int64 division with rounding
// 5. iterative while loop: multiply by 3 to normalize up
// 6. two clamp checks (branches)
// 7. pack back to int32 + int8
```

~40 instructions per element. Called 5120 times per RMSNorm.

Profile results (n=2560):
- mtfp21_mul: 17 ns/element (vs ~0.3 ns for float, vs ~0.03 ns for sign_epi8)
- mtfp21_add: 7.6 ns/element
- Total RMSNorm: 180 us (69x float32)
- div+rsqrt: 0.2 us (0.1% of total — not the bottleneck)

## What Went Wrong

We designed MTFP21 as a number system — mantissa × 3^exponent — with full
normalization after every operation. This is how IEEE 754 works. It's also
why IEEE 754 has a dedicated hardware unit (the FPU) instead of being
implemented in integer software.

We reimplemented floating-point arithmetic in software, in a non-standard
base, without hardware support. Every operation pays the normalization tax
that the FPU handles in silicon for IEEE 754.

The ternary AVX kernel works because it DOESN'T normalize. It operates on
raw trit vectors. sign_epi8 is a native instruction. add_epi8 is a native
instruction. No normalization, no exponent alignment, no widening until the
end. The hardware does the work.

MTFP21 as implemented does the OPPOSITE of what Shirley says to do.
Shirley says: let the hardware do what it does natively. Route computation
to the substrate that handles it best. The CPU handles ADD and MUL natively
via AVX2 integer. The iGPU handles EXP and LOG natively via hardware
transcendental units.

Instead, we built a software floating-point library that runs everything
on the CPU in scalar int64. We didn't route to the right substrate.
We didn't use the native instructions. We built an abstraction that fights
the hardware.

## What the RMSNorm Pipeline Actually Needs

RMSNorm: y[i] = x[i] * rsqrt(mean(x^2) + eps)

Breaking this into Shirley's six primes:
- x^2:          MUL(x, x)           → CPU native (integer multiply)
- sum(x^2):     ADD (accumulate)     → CPU native (integer add)
- mean:         MUL(sum, 1/n)        → CPU native (multiply by precomputed reciprocal)
- mean + eps:   ADD(mean, eps)       → CPU native
- rsqrt:        EXP(-0.5 * LOG(x))  → iGPU native (hardware V_EXP_F32, V_LOG_F32)
                OR integer LUT+NR    → CPU (validated, 0.2 us, not the bottleneck)
- y[i] = x[i] * scale: MUL          → CPU native (integer multiply by scalar)

Every per-element operation is CPU-native ADD or MUL. The only transcendental
(rsqrt) is called ONCE and takes 0.2 us. The entire pipeline should run at
near-AVX-integer speed for the per-element work, with one rsqrt call that's
either a fast CPU LUT+NR or an iGPU dispatch.

## The Right Architecture

The per-element operations should use the SAME approach as the ternary dot
product kernel: operate on raw integer vectors via AVX2. No MTFP21 wrapper.
No per-element normalization. No int64 widening.

```
Input:  int8 activations x[0..2559] (from matmul, range [-80, +80])

Step 1: Square — int8 × int8 → int16
        AVX2: _mm256_mullo_epi16 after widening, or _mm256_maddubs_epi16
        32 elements per cycle. Native.

Step 2: Accumulate squares — int16 → int32
        AVX2: _mm256_madd_epi16 to pair-add, then horizontal sum
        Result: one int32 sum of squares. Native.

Step 3: Compute scale = rsqrt(sum / n + eps)
        Option A: Integer LUT+NR (validated, 0.2 us)
        Option B: Dispatch to iGPU V_EXP_F32(-0.5 * V_LOG_F32(mean + eps))
        Option C: One float rsqrt (sqrtf — 5 ns, the pragmatic choice)
        Result: one float or fixed-point scale factor.

Step 4: Scale — int8 × scale → int8
        If scale is float: multiply, round, clamp to output range.
        If scale is fixed-point: integer multiply, shift, clamp.
        AVX2: _mm256_mullo_epi16 + shift + pack.
        32 elements per cycle. Native.
```

Total expected time for n=2560:
- Step 1: ~80 elements/cycle × ~3 GHz = ~32 cycles = ~10 ns
- Step 2: ~32 cycles = ~10 ns
- Step 3: 0.2 us (LUT+NR) or 5 ns (float rsqrt)
- Step 4: ~32 cycles = ~10 ns
- Total: ~0.2 us (LUT) or ~35 ns (float rsqrt)

vs current MTFP21: 180 us. That's 900x faster with LUT, 5000x with float rsqrt.

## MTFP21's Real Role

MTFP21 is not the per-element compute format. It's the TRANSPORT and STORAGE
format for values that need dynamic range — weight scales, normalization
parameters, accumulated statistics. Values computed once and applied many times.

The per-element work stays in raw integer (int8/int16/int32) via AVX2.
MTFP21 handles the scalar bookkeeping between the integer compute stages.

## Connection to the Six Primes and iGPU

The Ryzen 5 PRO 5675U has:
- CPU: AVX2 integer pipeline — ADD, MUL, MAX native at 32 elements/cycle
- iGPU: RDNA integrated graphics — V_EXP_F32, V_LOG_F32 at ~4 cycles

RMSNorm on the six primes:
```
x^2         → MUL  → CPU AVX2 (native, vectorized)
sum(x^2)    → ADD  → CPU AVX2 (native, vectorized)
1/n         → CONST → precomputed
mean + eps  → ADD  → CPU scalar
rsqrt(mean) → EXP(-0.5 * LOG(mean))  → iGPU V_EXP_F32 + V_LOG_F32
              OR LUT+NR               → CPU (0.2 us, already validated)
x * scale   → MUL  → CPU AVX2 (native, vectorized)
```

The iGPU earns its place if:
1. The rsqrt dispatch overhead (PCIe/shared-memory transfer) is < 0.2 us, AND
2. The iGPU can do rsqrt faster than the CPU LUT+NR

On this Ryzen with shared memory (no PCIe transfer), the dispatch overhead
is primarily the kernel launch latency. For a single rsqrt call, this is
likely higher than the 0.2 us CPU path. But for BATCHED rsqrt across
multiple layers (120 RMSNorms per forward pass), a single iGPU dispatch
computing all 120 rsqrts in parallel could be faster than 120 sequential
CPU LUT+NR calls.

This is the substrate routing the Shirley thesis predicts: CPU handles the
vectorized per-element work, iGPU handles the transcendentals in batch.
