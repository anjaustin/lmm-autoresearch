# Session 002 — Ground Truth: Real Integer Tests + Architecture Audit

**Date:** 2026-04-01
**Branch:** autoresearch/mar31-arc1
**Goal:** Validate simulation findings with real integer compute. Audit exact model architecture.

---

## Architecture Audit: BitNet b1.58-2B-4T (build_bitnet_158)

The model uses ReLU-squared, NOT SiLU. The "SiLU quantization has zero effect" finding
from Session 1 was testing a code path THAT DOES NOT EXIST in this model.

### Exact per-layer computation graph:

```
 #  Operation                    Domain    Ternary?  Tested?
──  ───────────────────────────  ────────  ────────  ────────
 1  Input from previous layer    float32   no        -
 2  RMSNorm (attn)               float32   rsqrt     SIM ONLY
 3  matmul: wq × norm            INT       yes       REAL (RANGE test)
 4  matmul: wk × norm            INT       yes       REAL (RANGE test)
 5  matmul: wv × norm            INT       yes       REAL (RANGE test)
 6  RoPE on Q                    float32   sin/cos   UNTESTED
 7  RoPE on K                    float32   sin/cos   UNTESTED
 8  Q @ K^T (attention scores)   float32   NO — float×float  UNTESTED
 9  softmax(scores)              float32   exp/sum   UNTESTED
10  attn_weights @ V             float32   NO — float×float  UNTESTED
11  RMSNorm (attn sub)           float32   rsqrt     SIM ONLY
12  matmul: wo × sub_norm        INT       yes       REAL (RANGE test)
13  wo_scale multiply            float32   no        UNTESTED
14  Residual ADD (attn)          float32   exact     UNTESTED
15  RMSNorm (FFN)                float32   rsqrt     SIM ONLY
16  matmul: ffn_gate × norm      INT       yes       REAL (RANGE test)
17  matmul: ffn_up × norm        INT       yes       REAL (RANGE test)
18  ReLU(gate)                   float32   MAX only  UNTESTED
19  ReLU(gate)^2                 float32   MUL only  UNTESTED
20  gate * up (element-wise)     float32   MUL       UNTESTED
21  RMSNorm (FFN sub)            float32   rsqrt     SIM ONLY
22  matmul: ffn_down × sub_norm  INT       yes       REAL (RANGE test)
23  ffn_down_scale multiply      float32   no        UNTESTED
24  Residual ADD (FFN)           float32   exact     UNTESTED
```

Plus at model boundaries:
```
 0  Embedding lookup             float32   -         UNTESTED
25  Final RMSNorm                float32   rsqrt     SIM ONLY  
26  LM head matmul (tok_embd)    Reuses embedding weights      UNTESTED
```

### Critical corrections from audit:

1. **The model uses ReLU-squared, NOT SiLU.** Our "SiLU is free" finding is INVALID
   for this model. ReLU-squared (MAX + MUL) is simpler than SiLU (requires sigmoid/exp),
   and is entirely composed of CPU-native primes. Good news for ternary, but the
   simulation tested the wrong operation.

2. **Attention Q@K^T and attn@V are NOT ternary matmuls.** These are float×float
   operations. Q, K, V are float32 after RoPE. The attention mechanism operates entirely
   in float32. Only the projections (wq, wk, wv, wo) are ternary matmuls.

3. **There are 4 RMSNorms per layer**, not 2:
   - attn_norm (before QKV)
   - attn_sub_norm (before wo)
   - ffn_norm (before gate/up)
   - ffn_sub_norm (before down)

4. **RoPE uses sin/cos** — transcendental operations applied AFTER the QKV projection.
   These are on Q and K only, not V. They're position-dependent and can be precomputed.

5. **Scale factors** (wo_scale, ffn_down_scale) are learned per-layer scalars.
   Element-wise multiply by a constant — trivial in any precision.

---

## Experiment: Real 4-trit integer activation quantization

Modified quantize_row_i8_s to clamp to [-40, +40] instead of [-127, +127].
This is REAL integer quantization — the int8 values flowing into the matmul
are clamped to 81 levels. No float simulation.

**PPL: 18.9055** (+0.05 from baseline) — LOOKS fine.

**Generation: LOOPS on some prompts.** A/B comparison with same seed:
- Baseline (127): diverse, develops ideas
- 4-trit (40): repetitive loops

**Generation quality sweep:**
| RANGE | Levels | Generation |
|-------|--------|------------|
| 40 | 81 | Loops |
| 60 | 121 | Coherent but terse |
| 80 | 161 | Good — matches baseline |
| 100 | 201 | Slight echoes |
| 127 | 255 | Baseline |

**CRITICAL FINDING: The simulation was unfaithful.** Float32 quantize-dequantize
predicted 4-trit was fine. Real integer quantization shows generation quality
needs RANGE=80 (~5 trits, 161 levels). Perplexity is blind to distribution
shape distortion.

---

## Revised atomic status

### REAL INTEGER (validated):
- Activation quantization: RANGE=80 (161 levels, ~5 trits) is floor for generation quality
- Ternary matmul kernel: sign_epi8 = maddubs, mathematically identical

### SIMULATION ONLY (unvalidated — may be wrong like the 4-trit finding):
- RMSNorm output quantization (7 trits claimed — but is this also optimistic?)
- Matmul output quantization (all the trit sweep results)

### WRONG:
- "SiLU is free" — model doesn't use SiLU. Uses ReLU-squared.

### UNTESTED:
- RoPE (sin/cos — precomputable)
- Softmax (exp/sum — in attention)
- Attention matmuls (float×float, not ternary)
- Residual ADDs
- Element-wise multiplies (scale factors, gate×up)
- Embedding lookup
- LM head
- ReLU-squared
