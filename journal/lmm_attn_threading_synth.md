# SYNTHESIS: Attention Multi-Threading — Split into Graph Nodes

## The Insight

Stop trying to synchronize threads within a single custom op. Split the attention into multiple graph nodes. Let ggml manage the thread lifecycle between nodes — that's what it does.

The spin-wait pattern is fundamentally fragile because it keeps non-zero threads ALIVE inside the callback while ggml expects them to be available for the next node. Whether ggml has inter-node barriers or not, the pattern creates undefined behavior when threads don't exit cleanly.

## The Plan

Replace one attention custom op with three:

```
Graph node: shirley_attn_prep (n_tasks=1)
  Thread 0: input conv → attn_norm → block-align
  Writes: shared act_mant, block_exp into per-layer params
  
Graph node: shirley_attn_qkv (n_tasks=MAX)
  ALL threads: Q matmul (partition rows), K matmul, V matmul
  Reads: shared act_mant from prep
  Writes: shared Q, K, V into per-layer params
  No synchronization needed — each thread writes its own rows

Graph node: shirley_attn_body (n_tasks=1)
  Thread 0: RoPE → cache store → Q@K^T → softmax → attn@V
            → sub_norm → block-align for wo
  Reads: shared Q, K, V from qkv
  Writes: shared wo_act, wo_bexp

Graph node: shirley_attn_wo (n_tasks=MAX)
  ALL threads: wo matmul (partition rows)
  Reads: shared wo_act from body
  Writes: shared wo_out

Graph node: shirley_attn_finish (n_tasks=1)
  Thread 0: residual ADD → output
  Reads: shared wo_out
  Writes: ggml output tensor
```

Five custom ops per layer instead of one. Each custom op is simple — no barriers, no phase counters, no spin-waits. Each thread enters, does its work, exits. ggml manages everything between ops.

## Why This Is Better

1. **Correct by construction.** No synchronization bugs possible — ggml handles all barriers.
2. **Each thread exits cleanly.** No thread is trapped in a spin-wait.
3. **The matmul ops can use the full thread pool.** No threads wasted on spin-waiting during sequential phases.
4. **Profile-friendly.** Each graph node's timing is visible to ggml's own profiler.

## Cost

5 graph nodes per layer × 30 layers = 150 additional nodes (from 60 to 210 total). The ggml graph node overhead is small (pointer chasing + dispatch). The benefit (correct multi-threading of 63% of compute) far exceeds the overhead.

## What About the FFN?

The FFN threading WORKS with the current pattern. Don't break what works. Optionally, refactor the FFN to the same split-node pattern for consistency, but it's not required.

## Implementation Order

1. Create 5 callback functions: prep, qkv, body, wo, finish
2. All share the per-layer params struct (userdata)
3. Wire into build_bitnet_158: 5 ggml_map_custom1 calls per layer
4. Test: all 3 prompts that caught the bug
5. Profile: compare speed vs single-threaded attention
