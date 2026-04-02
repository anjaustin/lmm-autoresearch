# NODES: Full MTFP21 LM Head

## Node 1: Shape change is the constraint
The LM head transforms [n_embd=2560] → [vocab_size=128256]. ggml custom ops dup the input shape. This is the ONLY reason the LM head is float.

## Node 2: ggml_mul_mat creates the right shape
`ggml_mul_mat(tok_embd, cur)` produces [vocab_size, n_tokens]. The shape creation logic is correct. What we need is the shape WITHOUT the float compute.

## Node 3: Tensor shape is mutable before graph execution
ggml tensors are C structs. ne[], nb[] are writable fields. The graph allocator reads them at allocation time. If I modify ne[] after custom op creation but before allocation, the allocator should see the correct shape.

## Node 4: ggml_map_custom2 output = dup(a)
If a is a [vocab_size, n_tokens] tensor, the output is also [vocab_size, n_tokens]. I need a tensor with that shape as `a`. I can create one with ggml_new_tensor_2d.

## Node 5: The dummy tensor approach
Create a dummy [vocab_size, n_tokens] tensor. Use it as `a` in ggml_map_custom2. The output has the right shape. The callback ignores `a` (it's a shape donor) and reads `b` (the normed output). The MTFP21 embedding table comes via userdata.

## Node 6: n_tokens is dynamic
The number of tokens changes per call (1 for generation, N for prompt eval). The dummy tensor needs the right n_tokens dimension. In build_bitnet_158, `n_tokens = this->n_tokens` is available.

## Node 7: Memory — 1.6 GB for MTFP21 embeddings
128256 × 2560 × 5 bytes = 1.6 GB. The machine has enough RAM but this is a significant allocation. Could be deferred or done incrementally.

## Node 8: The bypass option
Skip ggml entirely for the output stage. Allocate our own buffer, run the MTFP21 matmul, write float logits into ggml's output tensor data pointer. But this requires understanding ggml's memory management — will ggml free our buffer?

## Node 9: Performance — 128K dot products in MTFP21
Each logit requires a 2560-element MTFP21 dot product. 128256 logits = 128256 × 2560 MTFP21 muls + adds. At ~30ns per MTFP21 mul, that's ~10 seconds per token for the LM head alone. This is slower than the attention. But it runs once per token, not once per layer.

## Tensions:
- Node 1 vs Node 5: The dummy tensor approach solves the shape problem but feels hacky.
- Node 7 vs Node 9: The memory cost is high AND the compute is slow. Both need optimization eventually but correctness comes first.
- Node 4 vs Node 6: The dynamic n_tokens means the shape donor tensor must be created at graph build time, not init time.
