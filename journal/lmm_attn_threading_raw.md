# RAW: Attention Multi-Threading — Why It Keeps Breaking

The FFN multi-threading works. The attention multi-threading doesn't. I've tried three approaches and all corrupt the output. I need to stop trying variations of the same idea and understand what's actually happening.

What I know:
- ggml calls the custom op callback with (ith, nth) for each thread
- The FFN uses ad-hoc phase counters (mt_phase) with spin-waits and it works
- The same pattern applied to attention corrupts output on prompts 2 and 3
- A ggml-style barrier (atomic counter + spin) also corrupts
- A monotonic counter also corrupts
- The first prompt ("Paris") often works; longer generation fails
- The corruption is repetitive text — not garbage, but the model repeating itself

What I think is happening but haven't verified:
- ggml thread pool reuses threads across custom op calls (layer boundaries)
- Threads from layer N's attention haven't fully exited when layer N+1 starts
- The shared params struct (mt_phase, mt_act, mt_qkv) gets written by the wrong layer's threads

What scares me:
- The FFN uses the EXACT SAME pattern and works. Why would attention be different?
- If the thread lifecycle IS clean between custom ops, then the bug is in my code, not in ggml's threading
- I might be chasing a synchronization ghost when the real issue is something mundane

What's probably wrong with my first instinct:
- I keep assuming the problem is cross-call races. But the first prompt works. If it were a cross-call race, even the first prompt should fail (it goes through 30 layers).
- The corruption is REPETITIVE text, not random garbage. This suggests the model is computing something, just the wrong thing. Maybe the KV cache is getting corrupted?

Wait — the KV cache. The attention writes to `p->k_cache_mant`, `p->k_cache_exp` at position `pos`. If multiple threads are in the attention body (thread 0 writes cache, while a stale thread from a previous call also writes cache at a different position), the cache gets corrupted. But only thread 0 executes the attention body... unless the spin-wait at the end doesn't hold.

Actually, a different thought: the VLAs. The attention body uses massive VLAs on thread 0's stack: `mtfp21_t q_rot[2560]`, `mtfp21_t attn_out[2560]`, `mtfp21_t scores[kv_len]`. Each mtfp21_t is 8 bytes. With kv_len growing per token: at token 20, `scores[20]` is fine. But `q_rot[2560]` = 20 KB, `attn_out[2560]` = 20 KB, `k_rot[640]` = 5 KB, plus the block-align VLAs. Total stack per token: ~60 KB.

When threading is enabled, ggml creates worker threads with their own stacks. Thread 0 might be the MAIN thread with a large stack, or it might be a worker thread with a limited stack. If the VLAs overflow the worker thread's stack, we get undefined behavior — which could look like repetitive output if the stack corruption is semi-consistent.

The FFN doesn't have this problem because thread 0's sequential work uses smaller VLAs (the FFN trivials are in n_ff=6912 but as int16/int8, not full mtfp21_t structs).

Three open questions:
1. Is the issue stack overflow on worker threads from VLAs?
2. Is the issue KV cache corruption from stale threads?
3. Is the issue something else entirely that I'm not seeing?
