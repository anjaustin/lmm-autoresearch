# RAW: Full MTFP21 LM Head — The Last Float Operation

The LM head is the one remaining float32 matmul in the Shirley pipeline. Everything else — 30 layers of attention and FFN, output norm — runs in MTFP21. The LM head takes the normalized output [n_embd=2560] and produces logits [vocab_size=128256] via a matmul against the embedding table (tied weights).

Why is it float? Two reasons I've been telling myself:
1. The embedding table is float32 (not ternary)
2. The ggml custom op system can't produce a tensor with a different shape than the input

Reason 1 is solved — I wrote the conversion code. The embedding table CAN be MTFP21. The allocation is 1.6 GB, which is large but this is a 2B parameter model already using ~1.1 GB for weights. Another 1.6 GB for the MTFP21 embedding table is feasible on this machine (it has enough RAM).

Reason 2 is the real blocker. ggml_map_custom1 creates output as dup(input). ggml_map_custom2 also creates output as dup(a). The LM head changes dimensionality: [2560] → [128256]. No ggml custom op can express this shape change.

What scares me: I might need to bypass ggml entirely for the output stage. Create my own output tensor, do the MTFP21 matmul into it, and hand the float logits back to ggml's sampling infrastructure. This means understanding how ggml's sampling reads the logits tensor — where it expects the data, what shape, what type.

What's probably wrong with my first instinct: I keep trying to work WITHIN ggml. But we already bypassed ggml for the attention and FFN. The output stage is simpler than either of those. It's one matmul and a conversion to float.

The embedding lookup at the input is similar — it's a table lookup that produces a different shape. But it's already handled by ggml (llm_build_inp_embd). The output is the reverse: model internal → discrete tokens.

What if I just... write the logits directly into the ggml output tensor's data buffer? The ggml_mul_mat creates a tensor with the right shape [vocab_size, n_tokens]. I can keep that for the shape/allocation, but override the data with MTFP21-computed logits in a custom op that runs BEFORE the mul_mat uses the tensor. Or — use ggml_mul_mat for the shape, then replace its compute.

Actually: ggml_map_custom2(a, b) creates output shaped like a. If a = a tensor I create with shape [vocab_size, n_tokens]... I can create a dummy tensor with the right shape and use it as a. The custom2 op receives b (the normed output) and the userdata (which has the MTFP21 embedding table). The output tensor has the right shape.

Wait — I can create a tensor with ggml_new_tensor_2d(ctx, GGML_TYPE_F32, vocab_size, n_tokens). Then use ggml_map_custom2_inplace(shape_tensor, normed_output, lmhead_fn, ...). The output IS shape_tensor, which has the right dimensions. The custom2 callback writes logits into dst->data.

This might actually work. The shape_tensor needs to be a graph node so ggml allocates memory for it. I can make it depend on the normed output so it executes in the right order.

Or even simpler: just use ggml_new_tensor to create [vocab_size, n_tokens], set it as the output of a custom op... but custom ops create their own output via dup.

The real question: can I mutate the output tensor's ne[] after ggml_map_custom2 creates it? Before graph execution, the tensor is just a data structure. If I change ne[0] and nb[0] after the custom2 call but before ggml_build_forward_expand... would the allocator see the right shape?

Three open questions:
1. Does ggml_gallocr allocate based on ne[] at allocation time or at creation time?
2. Can I create a properly-shaped tensor and assign a custom op to it manually?
3. Is there a simpler way I'm not seeing?
