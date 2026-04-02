# SYNTHESIS: Removing ggml from the Shirley pipeline

## What ggml does for us (8 calls)

1. `ggml_new_graph_custom` — allocates the computation graph
2. `llm_build_inp_embd` — embedding lookup (float32 table → float32 vector)
3. `ggml_map_custom1` × 3 — our custom ops (attention, FFN, output norm)
4. `build_inp_out_ids` + `ggml_get_rows` — last-layer output token selection
5. `llm_build_lora_mm` — LM head matmul (float32)
6. `ggml_build_forward_expand` — finalizes the graph for execution

## What we actually need

The execution is sequential. No graph scheduling needed.

```
For each inference call:
  1. Read token IDs from the batch
  2. Look up embeddings (read from model data)
  3. Convert to MTFP21
  4. For each layer: attention(MTFP21) → FFN(MTFP21)
  5. Output norm (MTFP21)
  6. LM head (MTFP21 × MTFP21 → float logits)
  7. Return logits to sampling
```

Steps 3-5 are already Shirley code. Steps 1, 2, 6, 7 are the ggml dependency.

## The replacement

### Embedding lookup
Read directly from `model.tok_embd->data`. The tensor data IS the embedding table — a contiguous float32 array. For each token ID, the embedding is at offset `token_id × n_embd`. Convert to MTFP21 immediately.

The memory access issue (segfault at ~75%) was caused by trying to read ALL 128K rows at once during the lazy conversion loop. The embedding lookup only reads ONE row per token — this will not hit the access boundary.

### LM head
The LM head is the reverse embedding lookup: `logit[v] = Σ normed[d] × embd[v][d]`. This reads all 128K rows — which is where the segfault occurs. But we don't need to read them all at once. We can compute one row at a time, reading incrementally. If the memory mapping faults at a specific boundary, we can detect and handle it.

Alternative: use ggml's own `ggml_mul_mat` for the shape and let it handle the memory access. Keep this ONE ggml call if the memory access issue can't be resolved.

### Token selection
`build_inp_out_ids` + `ggml_get_rows` selects which tokens to output on the last layer. For generation, this is always the last token. For prompt eval with `n_outputs < n_tokens`, it selects specific positions. We can implement this as a simple index operation.

### Graph infrastructure
Not needed. Our execution is a simple loop. No graph.

## The plan

Phase A: Write `shirley_forward()` — a standalone function that takes token IDs and returns float logits. Calls our custom ops directly, no ggml graph. The embedding lookup and token selection are inline.

Phase B: The LM head either uses the row-at-a-time approach (MTFP21 dot products, one vocab row per iteration) or falls back to ggml_mul_mat.

Phase C: Wire `shirley_forward()` into llama.cpp's decode path, replacing `build_bitnet_158()` entirely.
