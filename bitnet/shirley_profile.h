/*
 * shirley_profile.h — lightweight profiling for Shirley ops
 */
#ifndef SHIRLEY_PROFILE_H
#define SHIRLEY_PROFILE_H

#include <time.h>
#include <stdio.h>

static inline double _shirley_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* Accumulate timings across calls, print summary after N tokens */
static struct {
    double attn_input_conv;
    double attn_norm;
    double attn_qkv_matmul;
    double attn_rope;
    double attn_kv_cache;
    double attn_qk_dot;
    double attn_softmax;
    double attn_av;
    double attn_sub_norm;
    double attn_wo;
    double attn_residual;
    double ffn_input_conv;
    double ffn_norm;
    double ffn_gate_up;
    double ffn_trivials;
    double ffn_sub_norm;
    double ffn_down;
    double ffn_residual;
    int tokens;
} _sp = {0};

static inline void _shirley_profile_print(void) {
    if (_sp.tokens == 0) return;
    double total = _sp.attn_input_conv + _sp.attn_norm + _sp.attn_qkv_matmul +
        _sp.attn_rope + _sp.attn_kv_cache + _sp.attn_qk_dot + _sp.attn_softmax +
        _sp.attn_av + _sp.attn_sub_norm + _sp.attn_wo + _sp.attn_residual +
        _sp.ffn_input_conv + _sp.ffn_norm + _sp.ffn_gate_up +
        _sp.ffn_trivials + _sp.ffn_sub_norm + _sp.ffn_down + _sp.ffn_residual;

    fprintf(stderr, "\n=== SHIRLEY PROFILE (%d tokens, %.1f ms total) ===\n", _sp.tokens, total*1000);
    fprintf(stderr, "ATTENTION:\n");
    fprintf(stderr, "  input_conv:   %6.1f ms (%4.1f%%)\n", _sp.attn_input_conv*1000, _sp.attn_input_conv/total*100);
    fprintf(stderr, "  norm:         %6.1f ms (%4.1f%%)\n", _sp.attn_norm*1000, _sp.attn_norm/total*100);
    fprintf(stderr, "  QKV matmul:   %6.1f ms (%4.1f%%)\n", _sp.attn_qkv_matmul*1000, _sp.attn_qkv_matmul/total*100);
    fprintf(stderr, "  RoPE:         %6.1f ms (%4.1f%%)\n", _sp.attn_rope*1000, _sp.attn_rope/total*100);
    fprintf(stderr, "  KV cache:     %6.1f ms (%4.1f%%)\n", _sp.attn_kv_cache*1000, _sp.attn_kv_cache/total*100);
    fprintf(stderr, "  Q@K^T:        %6.1f ms (%4.1f%%)\n", _sp.attn_qk_dot*1000, _sp.attn_qk_dot/total*100);
    fprintf(stderr, "  softmax:      %6.1f ms (%4.1f%%)\n", _sp.attn_softmax*1000, _sp.attn_softmax/total*100);
    fprintf(stderr, "  attn@V:       %6.1f ms (%4.1f%%)\n", _sp.attn_av*1000, _sp.attn_av/total*100);
    fprintf(stderr, "  sub_norm:     %6.1f ms (%4.1f%%)\n", _sp.attn_sub_norm*1000, _sp.attn_sub_norm/total*100);
    fprintf(stderr, "  wo matmul:    %6.1f ms (%4.1f%%)\n", _sp.attn_wo*1000, _sp.attn_wo/total*100);
    fprintf(stderr, "  residual:     %6.1f ms (%4.1f%%)\n", _sp.attn_residual*1000, _sp.attn_residual/total*100);
    fprintf(stderr, "FFN:\n");
    fprintf(stderr, "  input_conv:   %6.1f ms (%4.1f%%)\n", _sp.ffn_input_conv*1000, _sp.ffn_input_conv/total*100);
    fprintf(stderr, "  norm:         %6.1f ms (%4.1f%%)\n", _sp.ffn_norm*1000, _sp.ffn_norm/total*100);
    fprintf(stderr, "  gate+up:      %6.1f ms (%4.1f%%)\n", _sp.ffn_gate_up*1000, _sp.ffn_gate_up/total*100);
    fprintf(stderr, "  trivials:     %6.1f ms (%4.1f%%)\n", _sp.ffn_trivials*1000, _sp.ffn_trivials/total*100);
    fprintf(stderr, "  sub_norm:     %6.1f ms (%4.1f%%)\n", _sp.ffn_sub_norm*1000, _sp.ffn_sub_norm/total*100);
    fprintf(stderr, "  down:         %6.1f ms (%4.1f%%)\n", _sp.ffn_down*1000, _sp.ffn_down/total*100);
    fprintf(stderr, "  residual:     %6.1f ms (%4.1f%%)\n", _sp.ffn_residual*1000, _sp.ffn_residual/total*100);
    fprintf(stderr, "================================================\n");
}

#define SP_START double _t0 = _shirley_now()
#define SP_LAP(field) do { double _t1 = _shirley_now(); _sp.field += (_t1 - _t0); _t0 = _t1; } while(0)
#define SP_TOKEN() do { _sp.tokens++; if (_sp.tokens == 10) _shirley_profile_print(); } while(0)

#endif
