# Kernel Extraction Plan

## Source

The SSTT project (`~/Projects/000-research/sstt/`) contains validated, production-ready ternary computation kernels implemented in C with AVX2 SIMD. These kernels implement four of the six primes natively and provide the routing infrastructure for Shirley.

## Kernels to Extract

### Tier 1: Prime Implementations (Required)

#### K1. Ternary MUL — `sign_epi8` dot product
**Source:** `sstt_v2.c:272-289`
**What it does:** 32-wide ternary multiplication via `_mm256_sign_epi8`, accumulated into a dot product.
**Maps to prime:** MUL + ADD (fused multiply-accumulate)
**Why it matters:** This is the fundamental compute operation for both inference and routing. 800 ternary products in ~25 cycles.

```c
static inline int32_t ternary_dot(const int8_t *a, const int8_t *b) {
    __m256i acc = _mm256_setzero_si256();
    for (int i = 0; i < PADDED; i += 32) {
        __m256i va = _mm256_load_si256((const __m256i *)(a + i));
        __m256i vb = _mm256_load_si256((const __m256i *)(b + i));
        __m256i prod = _mm256_sign_epi8(va, vb);
        acc = _mm256_add_epi8(acc, prod);
    }
    // horizontal reduction to int32
}
```

**Extraction notes:** Parameterize `PADDED` dimension. Add horizontal sum variants (epi8 → epi16 → epi32 widening for large accumulations without overflow).

---

#### K2. Ternary Quantization — 3-level AVX2
**Source:** `sstt_v2.c:229-266`
**What it does:** Quantizes continuous values to {-1, 0, +1} using dual-threshold comparison, 32-wide.
**Maps to prime:** This is the domain entry point — continuous → ternary.
**Why it matters:** Every input enters the ternary domain through this gate. It's the precision boundary.

```c
__m256i spx = _mm256_xor_si256(px, bias);         // unsigned → signed comparison
__m256i pos = _mm256_cmpgt_epi8(spx, thi);        // above upper threshold?
__m256i neg = _mm256_cmpgt_epi8(tlo, spx);        // below lower threshold?
__m256i p = _mm256_and_si256(pos, one);
__m256i n = _mm256_and_si256(neg, one);
__m256i result = _mm256_sub_epi8(p, n);            // {-1, 0, +1}
```

**Extraction notes:** Parameterize thresholds (currently hardcoded to 85/170 for pixel data). Add float → trit variant for continuous-domain outputs returning to ternary after EXP/LOG.

---

#### K3. Ternary MAX — `max_epi8`
**Source:** Used throughout SSTT, standard AVX2 intrinsic.
**What it does:** Element-wise maximum across 32 int8 lanes.
**Maps to prime:** MAX
**Why it matters:** Selection, ReLU (`max(x, 0)`), routing decisions.

```c
__m256i result = _mm256_max_epi8(a, b);
```

**Extraction notes:** Trivial — it's a single intrinsic. Wrap with argmax variant for routing (which lane is maximum).

---

#### K4. CONST — immediate loading
**Maps to prime:** CONST
**Why it matters:** Bias, thresholds, fixed values.
**Extraction notes:** Not really a kernel — it's `_mm256_set1_epi8(c)` for broadcast, or memory load for vectors. Include in the primitive API for completeness.

---

### Tier 2: Routing Infrastructure (Required)

#### K5. Block Encoding — trit tuple compression
**Source:** `sstt_v2.c`, `sstt_bytepacked.c:158-182`
**What it does:** Encodes N ternary trits into a single byte index. 3 trits → 27 values (0-26).
**Why it matters:** Converts ternary vectors into compact indices for signature matching and lookup tables.

```c
static inline uint8_t block_encode(int8_t t0, int8_t t1, int8_t t2) {
    return (uint8_t)((t0+1)*9 + (t1+1)*3 + (t2+1));
}
```

**Extraction notes:** Generalize to N-trit encoding. 3 trits → 27 values, 4 trits → 81 values, etc. Consider SIMD vectorization for batch encoding.

---

#### K6. Hamming Distance Matching — XOR + POPCNT
**Source:** Used across TriX and SSTT codebases. LCVDB has the most optimized implementation.
**What it does:** Computes Hamming distance between two ternary signatures.
**Why it matters:** This IS semantic routing. Input signature vs. shape signature → distance → route to closest match.

**Extraction notes:** Check LCVDB (`~/Projects/000-research/lcvdb/`) for the optimized version. SSTT's multi-probe Hamming-1 expansion (K7) builds on this.

---

#### K7. Multi-Probe Hamming-1 Neighbor Expansion
**Source:** `sstt_v2.c:553-582`
**What it does:** For a given trit tuple, enumerates all Hamming-distance-1 neighbors (each trit flipped to its two alternative values). Always exactly 6 neighbors for a 3-trit block.
**Why it matters:** Relaxes exact matching to fuzzy matching. In SSTT this added +2.38 percentage points of accuracy. In Shirley, this enables soft routing — an input can partially activate nearby shapes.

**Extraction notes:** Generalize to N-trit blocks with Hamming distance K.

---

#### K8. Hot Map Lookup Table
**Source:** `sstt_v2.c:595-660`, `sstt_fused.S:176-205`
**What it does:** Table mapping (position, block_value) → class scores. L2-cache resident.
**Why it matters:** This is the shape library in table form — precomputed outputs indexed by input pattern. In Shirley, the hot map becomes the frozen shape lookup: given a routing decision (which shape) and an input encoding (block value), retrieve the precomputed result.

**Extraction notes:** Generalize from classification (class scores) to arbitrary shape outputs. The structure — positional lookup table in L2 — is the key idea, not the specific contents.

---

#### K9. Information Gain Weighting
**Source:** `sstt_v2.c:484-551`
**What it does:** Computes per-position, per-channel importance weights from Shannon entropy. Closed-form, no learning.
**Why it matters:** Not all routing positions are equally informative. IG weighting tells you which positions matter most — center of a digit matters more than corners. In Shirley, this maps to routing priority: which parts of the input should most strongly influence shape selection.

**Extraction notes:** Generalize from image positions to arbitrary input dimensions.

---

### Tier 3: Transcendental Dispatch (Required for iGPU)

#### K10. EXP dispatch — iGPU compute shader
**Source:** New — to be implemented.
**What it does:** Dispatches `exp(x)` operations to the AMD iGPU's V_EXP_F32 hardware unit.
**Maps to prime:** EXP
**Interface:** Accept float32 buffer from CPU (shared memory), return float32 results.

**Implementation options:**
- OpenCL kernel targeting AMD iGPU
- ROCm HIP kernel (if supported on iGPU)
- Vulkan compute shader

**Extraction notes:** This is the domain crossing — ternary integer values on CPU become float32 for transcendental computation on iGPU, then return. The precision boundary management from b158 research applies here.

---

#### K11. LOG dispatch — iGPU compute shader
**Source:** New — to be implemented.
**What it does:** Dispatches `log(x)` operations to the AMD iGPU's V_LOG_F32 hardware unit.
**Maps to prime:** LOG
**Interface:** Same as K10 — shared memory buffer exchange.

**Extraction notes:** EXP and LOG share the same transcendental function unit on the iGPU. They can share dispatch infrastructure. Consider a combined transcendental dispatch kernel that handles both.

---

## Extraction Strategy

### Phase 1: CPU Primitives
Extract K1-K4 into a standalone header: `shirley/src/primes.h`
- Pure AVX2, no dependencies beyond `<immintrin.h>`
- Each prime as an inline function
- Scalar fallbacks for non-AVX2 platforms
- Exhaustive verification tests (like FLYNNCONCEIVABLE's 460,928 combinations)

### Phase 2: Routing Infrastructure
Extract K5-K9 into routing modules: `shirley/src/routing.h`, `shirley/src/shapes.h`
- Block encoding and signature matching
- Hot map structure (generalized from classification to arbitrary shape output)
- Hamming distance matching with multi-probe expansion
- IG weighting for routing priority

### Phase 3: Transcendental Dispatch
Implement K10-K11: `shirley/src/transcendental.h`, `shirley/src/dispatch_gpu.c`
- OpenCL or Vulkan compute shaders for iGPU
- Shared memory buffer management
- Domain crossing protocol (ternary → float32 → ternary)
- Fallback to CPU software EXP/LOG when iGPU unavailable

### Phase 4: Integration
Wire the primes, routing, and dispatch into a unified API: `shirley/src/shirley.h`
- Shape definition (composition of primes)
- Automatic substrate routing (static, from prime composition)
- Semantic routing (dynamic, from input-shape matching)
- Benchmark harness for LEMM loop

## Validation

Each extracted kernel must be verified against the source:

| Kernel | Verification method |
|--------|-------------------|
| K1 (ternary dot) | Exhaustive test: all 3^N × 3^N input pairs for small N, random sampling for large N |
| K2 (quantization) | Round-trip: quantize → verify trit values → compare to SSTT reference output |
| K3 (MAX) | Exhaustive: all 256 × 256 int8 pairs |
| K5 (block encode) | Exhaustive: all 27 trit triples → verify unique indices 0-26 |
| K6 (Hamming) | Exhaustive: all signature pairs for small dimension |
| K7 (multi-probe) | Verify neighbor count (always 6 for 3-trit), verify Hamming distance = 1 |
| K8 (hot map) | Load SSTT's trained hot map, reproduce SSTT classification accuracy |
| K10-K11 (EXP/LOG) | Compare iGPU output to CPU libm output, verify error < 1 ULP |

## File Structure

```
shirley/
├── docs/
│   ├── OVERVIEW.md
│   ├── SIX_PRIMES.md
│   ├── ROUTING.md
│   ├── TERNARY_VS_BINARY.md
│   ├── KERNEL_EXTRACTION.md
│   └── LEMM_APPLICATION.md
├── src/
│   ├── primes.h              ← K1-K4: six primes as inline functions
│   ├── routing.h             ← K5-K7: encoding, matching, multi-probe
│   ├── shapes.h              ← K8: hot map / shape lookup structure
│   ├── weights.h             ← K9: IG weighting, routing priority
│   ├── transcendental.h      ← K10-K11: iGPU dispatch interface
│   ├── dispatch_gpu.c        ← K10-K11: OpenCL/Vulkan implementation
│   ├── shirley.h             ← unified API
│   └── shirley.c             ← integration
├── tests/
│   ├── test_primes.c         ← exhaustive prime verification
│   ├── test_routing.c        ← routing correctness
│   └── test_dispatch.c       ← transcendental dispatch verification
├── bench/
│   └── benchmark.c           ← LEMM evaluation harness
├── program.md                ← LEMM agent instructions
├── results.tsv               ← experiment log (gitignored)
├── strategy_override.md      ← human steering (gitignored)
└── journal/                  ← session reasoning artifacts (gitignored)
```
