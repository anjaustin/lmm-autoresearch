# REFLECT: Full MTFP21 LM Head

## Why three times on Node 5 (dummy tensor approach):

**Why is it hacky?** Because the dummy tensor has no real data — it exists only to set the output shape. The custom op ignores it.

**Why does that matter?** Because ggml will allocate memory for the dummy tensor's data buffer even though nothing writes to it. Wasted memory. But vocab_size × n_tokens × sizeof(float) = 128256 × 1 × 4 = 500 KB per token. Negligible.

**Why am I hesitating?** Because I want a "clean" solution. But clean is the enemy of done. The FFN and attention custom ops also use patterns that ggml wasn't designed for. This is another one. The dummy tensor is a shape adapter — its purpose is explicit and its cost is trivial.

## Hidden assumptions:

1. **I assume the allocator uses ne[] at allocation time.** If it caches sizes at tensor creation, modifying ne[] later won't help. But Node 4 says ggml_map_custom2 creates the output via ggml_dup_tensor(a), which reads a->ne[] at that moment. So the shape is set at custom2 creation time, not later.

2. **I assume ggml's sampling reads from dst->data as a contiguous float array of [vocab_size × n_tokens].** This should be true — that's what ggml_mul_mat produces.

3. **I assume n_tokens is known at graph build time.** It is — `this->n_tokens` in the graph builder.

## What would this look like if it were easy?

Create a tensor with shape [vocab_size, n_tokens]. Assign a custom compute function to it. The function reads the normed output and the MTFP21 embedding table, writes float logits.

That's exactly what `ggml_map_custom2(shape_donor, normed_output, fn, 1, userdata)` does if shape_donor has shape [vocab_size, n_tokens].

## The simplest version that could work:

```c
// In build_bitnet_158, after output norm:
struct ggml_tensor * lm_shape = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_vocab, n_tokens);
cur = ggml_map_custom2(ctx0, lm_shape, cur, shirley_lmhead_compute, 1, &shirley_output_p);
```

The callback writes logits into dst->data. dst has shape [vocab_size, n_tokens] because it was duped from lm_shape. The normed output is in `b`. The MTFP21 embedding table is in userdata.

## Core insight in one sentence:

The shape donor tensor is the adapter between ggml's type system and Shirley's compute — it costs nothing and solves the constraint completely.

## Resolved tensions:
- Node 1 vs Node 5: Not hacky — it's a pattern. Shape donor tensors are used elsewhere in ggml (views, reshapes).
- Node 7: Defer the 1.6 GB allocation for now. Convert embedding on first use if memory allows, or convert in chunks.
- Node 9: 10 seconds per token for the LM head is acceptable for correctness validation. Optimization (AVX2, chunked compute) comes later.
