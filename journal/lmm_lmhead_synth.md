# SYNTHESIS: Full MTFP21 LM Head

## The Plan

Use a shape donor tensor to create a custom2 op with [vocab_size, n_tokens] output. The donor provides the shape; the callback provides the compute. Zero float in the LM head matmul.

## Implementation

### 1. In build_bitnet_158(), replace the LM head:

```c
// Create shape donor: [vocab_size, n_tokens]
struct ggml_tensor * lm_shape = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 
    hparams.n_vocab, n_tokens);

// MTFP21 LM head: normed output × MTFP21 embedding table → float logits
cur = ggml_map_custom2(ctx0, lm_shape, cur,
    shirley_lmhead_compute, 1, &shirley_output_p);
```

### 2. The callback (shirley_lmhead_compute):

```
dst: [vocab_size, n_tokens] — shape from donor
a: lm_shape (ignored — shape donor only)
b: normed output [n_embd, n_tokens] — the real input
userdata: shirley_output_params with MTFP21 embedding table

For each token:
  Convert normed[n_embd] to MTFP21
  For each vocab entry v:
    logit[v] = MTFP21 dot product(normed, embd[v])
    Convert to float
  Write float logits to dst
```

### 3. Enable embedding conversion at init:

Pass model.tok_embd to shirley_output_params_init. The 1.6 GB allocation happens once at model load.

### 4. What this achieves:

The full forward pass becomes:
```
Embedding lookup (float32 table → float32 → MTFP21 at layer 0 input)
→ 30 layers × [Shirley attention + Shirley FFN] (MTFP21)
→ Output norm (MTFP21)
→ LM head (MTFP21 matmul → float logits)
→ Sampling (float — the discrete boundary)
```

The ONLY float remaining: the embedding lookup (float32 table, converted to MTFP21 at first layer entry) and the final logit-to-float conversion for sampling. Both are at the boundary between discrete token IDs and continuous computation.
