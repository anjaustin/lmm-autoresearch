# REFLECT: Attention Multi-Threading

## Why three times on Node 8 (ggml inter-node barriers):

**Why do I believe ggml has barriers between nodes?** Because ggml's own multi-threaded ops (like ggml_mul_mat) use barriers within a single node, and the graph scheduler processes nodes sequentially. The ggml_compute_forward function dispatches one node at a time.

**Can I verify this?** Yes — the ggml source at line 3418 shows ggml_barrier. And the graph execution loop processes nodes one at a time. Each node's compute is called for all threads, with barriers between phases within a node. Between nodes, ggml's scheduler moves to the next node only after all threads have completed the current one.

**If barriers exist, why does my attention threading fail?** The bug MUST be inside a single call. Not cross-call.

## The hidden assumption I was wrong about

I assumed the corruption was from cross-call races (threads from layer N leaking into layer N+1). I spent three attempts fixing cross-call synchronization. But if ggml guarantees inter-node barriers, cross-call races are impossible. **I was solving the wrong problem.**

## What would this look like if it were easy?

If the bug is inside a single call, it's a DATA bug, not a SYNCHRONIZATION bug. The threads are correctly synchronized (barrier after QKV matmul, barrier after wo matmul). But the data they write or read is wrong.

Possible data bugs:
1. **shirley_gemv_mtfp16_part writes to wrong rows.** If the row partitioning is off-by-one, threads overwrite each other's output.
2. **The shared workspace (mt_qkv) is too small.** If Q+K+V don't fit in the allocated buffer, threads write past the end.
3. **mt_act is overwritten between QKV and wo.** Thread 0 overwrites mt_act during sub_norm block-align. If a non-zero thread is still reading mt_act from the QKV phase... but the barrier should prevent this.
4. **The attention body reads from mt_q/mt_k/mt_v which are pointers into mt_qkv.** If the allocation size is wrong, these pointers overlap.

Let me check #4: `mt_qkv = malloc((n_embd + kv_dim + kv_dim) * sizeof(mtfp21_t))`. That's (2560 + 640 + 640) × 8 = 30,720 bytes. mt_q = mt_qkv, mt_k = mt_q + 2560, mt_v = mt_k + 640. Total: 2560 + 640 + 640 = 3840 mtfp21_t values. The Q matmul writes 2560 rows into mt_q. The K matmul writes 640 rows into mt_k. The V matmul writes 640 rows into mt_v. No overlap. This is correct.

Let me check #1: `shirley_gemv_mtfp16_part` computes rows [r0, r1). r0 = ith * rows_per, r1 = min((ith+1) * rows_per, n_output). With 6 threads and 2560 rows: rows_per = ceil(2560/6) = 427. Thread 0: [0, 427), Thread 1: [427, 854), ..., Thread 5: [2135, 2560). These don't overlap. The writes are to dst[r0..r1-1]. No thread writes outside its partition.

BUT: the Q matmul writes to mt_q (n=2560 output rows). The K matmul writes to mt_k (kv_dim=640 output rows). The V matmul writes to mt_v (kv_dim=640 output rows). All three use the SAME act_mant (mt_act) as input — that's read-only, so it's safe.

Wait — the K and V matmuls each have only 640 rows. With 6 threads: rows_per = ceil(640/6) = 107. Thread 5: [535, 640). Thread 4: [428, 535). These are fine. But what if rows_per × nth > n_output? 107 × 6 = 642 > 640. Thread 5's r1 = min(642, 640) = 640. Correct.

I'm not finding a data bug. Let me think about what's ACTUALLY different between FFN and attention threading.

The FFN's mt_phase reset to 0: `__atomic_store_n(&p->mt_phase, 0, __ATOMIC_RELEASE)`. Non-zero threads wait: `while (mt_phase != 0)`. When the NEXT token's call starts, thread 0 sets mt_phase=1. Non-zero threads see mt_phase != 0 is false (it IS 0), so they exit the wait... wait, `mt_phase != 0` — they wait WHILE mt_phase is NOT zero. So they're waiting for mt_phase TO BE zero. When it's 0, they proceed.

But proceed to WHERE? They re-enter the token loop, hit `if (ith == 0) { prep } else { while (mt_phase < 1) }`. If thread 0 hasn't started the next iteration yet, mt_phase is still 0, so `mt_phase < 1` is true, and they spin. This is correct.

For attention: THE SAME PATTERN. If ggml guarantees all threads complete before the next node, then between layer N and layer N+1, all threads have exited the function. The params struct is clean. Thread 0 enters the new call, non-zero threads enter, everyone starts fresh.

So... why does it break?

## New hypothesis: ggml doesn't guarantee all threads complete for custom ops

The graph scheduler may move to the next node when THREAD 0 returns from the custom op. Non-zero threads are still in the spin-wait. The next node starts with those threads still spinning in the PREVIOUS node's callback.

This would explain why FFN works (the next node after FFN is the next layer's attention, which is a DIFFERENT custom op with DIFFERENT params) but attention fails (the next attention op REUSES the same threadpool, and stale threads from the previous layer's attention are still in the spin-wait).

## Core insight

**The issue is NOT synchronization. It's that non-zero threads NEVER EXIT the custom op callback until they're released.** With the phase/spin-wait pattern, non-zero threads are trapped in the callback's spin-wait between tokens. Thread 0 exits, ggml moves to the next node, but the non-zero threads are STILL INSIDE the previous callback.

The FFN works because the non-zero threads DO exit: after the last token's down matmul, thread 0 resets mt_phase to 0, and non-zero threads break out of `while (mt_phase != 0)`, reach the end of the token loop, and exit the function naturally.

But wait — the attention had the same pattern. Let me re-read the old attention code... it had `while (mt_phase != 0)` at the end. And the non-zero threads would exit the token loop and return from the function. Same as FFN.

I need to actually instrument and TEST whether non-zero threads exit the callback. Not assume.

## What to actually try

1. **Test hypothesis: add a print when each thread exits the callback.** See if all 6 threads exit before the next node starts.
2. **Test hypothesis: replace VLAs with heap allocation in the attention body.** Rule out stack issues.
3. **The simplest thing that could work: just split into graph nodes.** Three custom ops per layer. ggml handles barriers. No spin-waits, no phase counters, no ambiguity.

Option 3 is the cleanest. The LMM told me this in the first pass but I ignored it because it felt "ugly" (more graph nodes). It's not ugly — it's correct by construction.
