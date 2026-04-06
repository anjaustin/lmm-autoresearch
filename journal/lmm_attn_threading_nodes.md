# NODES: Attention Multi-Threading

## Node 1: FFN threading works, attention doesn't
Same pattern, same params struct layout, same ggml custom op mechanism. The difference must be in the CONTENT of the ops, not the synchronization.

## Node 2: The corruption is repetitive, not random
"uncertainty principle states that the uncertainty principle states that" — the model loops. This isn't memory corruption (which would produce garbage). It's the model confidently predicting the same tokens. This suggests the attention mechanism is producing degenerate attention patterns — attending to the same positions repeatedly.

## Node 3: First prompt often works
"Paris." is correct. The first token of generation works fine. The corruption appears during autoregressive generation, growing worse with token count. This correlates with KV cache growth.

## Node 4: VLA stack usage in attention is massive
Thread 0's attention body allocates ~60 KB of VLAs per token per layer call: q_rot[2560], k_rot[640], attn_out[2560], scores[kv_len], q_mant_arr[2560], q_exp_arr[2560], plus the sub_norm VLAs. With 30 layers, if ggml reuses the same thread for multiple layers, the stack might NOT be cleaned between calls — VLAs from the previous layer's attention are still on the stack when the next layer starts.

## Node 5: The FFN's thread 0 sequential work is lighter
FFN thread 0 sequential: inp_m[2560] (20 KB) + gate_mant16[6912] + up_mant16[6912] + sq_mant32[6912] + ffn_out[6912] ≈ 80 KB of VLAs. Actually this is larger than attention. So stack size isn't obviously the differentiator.

## Node 6: ggml worker thread stack size
ggml's threadpool creates pthreads. Default pthread stack is 8 MB on Linux. 60-80 KB of VLAs should be fine. Stack overflow is probably NOT the issue.

## Node 7: The mt_phase reset race (original diagnosis)
With the ad-hoc pattern: thread 0 sets mt_phase=0 at the end. Non-zero threads spin on `mt_phase != 0`. If ggml calls the next layer's attention before all threads exit the spin-wait, they see mt_phase=1 (from the new call), exit the wait, and proceed with STALE data or into the WRONG matmul.

But this same race exists in the FFN — `mt_phase=0` reset, `mt_phase != 0` wait. And FFN works. Unless ggml inserts a barrier between custom ops but not between layers...

## Node 8: ggml graph execution order
The graph has: attn[0] → ffn[0] → attn[1] → ffn[1] → ... → attn[29] → ffn[29]. Each is a separate graph node. ggml processes nodes sequentially. Between nodes, ggml MUST synchronize threads (otherwise its own ops would race). So there IS a barrier between custom op calls.

If there's a barrier between calls, then the mt_phase race CANNOT happen — all threads from call N have exited before call N+1 starts. Which means the bug is NOT a cross-call race.

## Node 9: The bug is inside a single call
If cross-call races are impossible (due to ggml inter-node barriers), the corruption must come from within a single attention call. The matmul output (mt_qkv) written by all threads, then read by thread 0 for RoPE and attention body. If any thread writes to the WRONG rows (indexing bug in shirley_gemv_mtfp16_part), the Q/K/V data is corrupted.

## Tensions:
- Node 1 vs Node 9: if it's an indexing bug, why does FFN work? Same partitioning function.
- Node 2 vs Node 4: repetitive output suggests attention degeneration, not generic corruption
- Node 7 vs Node 8: if ggml has inter-node barriers, the phase race shouldn't matter
- Node 3: first prompt works → the corruption accumulates over tokens, pointing to KV cache
