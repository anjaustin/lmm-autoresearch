#!/bin/bash
# eval_harness.sh — Fixed evaluation harness for BitNet ternary conversion
# DO NOT MODIFY — this is the ground-truth measurement
#
# Usage: ./eval_harness.sh [rebuild]
#   If "rebuild" is passed, recompiles before measuring.
#
# Outputs: perplexity (PPL), prompt eval speed (tok/s), peak RSS (MB)
# Exit code: 0 if valid, 1 if NaN/Inf/missing

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="$SCRIPT_DIR/models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf"
EVAL_DATA="$SCRIPT_DIR/eval_data/wikitext2_test.txt"
BUILD_DIR="$SCRIPT_DIR/build"
PERP_BIN="$BUILD_DIR/bin/llama-perplexity"
THREADS=6
CHUNKS=20
CTX=512
LOG="$SCRIPT_DIR/run.log"

# Rebuild if requested
if [[ "${1:-}" == "rebuild" ]]; then
    echo "Rebuilding..."
    cmake --build "$BUILD_DIR" --config Release > "$SCRIPT_DIR/build.log" 2>&1
    BUILD_EXIT=$?
    if [[ $BUILD_EXIT -ne 0 ]]; then
        echo "BUILD_FAILED"
        echo "build_status: FAILED" > "$LOG"
        exit 1
    fi
    echo "Build complete."
fi

# Run perplexity measurement
echo "Running perplexity measurement ($CHUNKS chunks, ctx=$CTX)..."
/usr/bin/time -v "$PERP_BIN" \
    -m "$MODEL" \
    -f "$EVAL_DATA" \
    -t "$THREADS" \
    --chunks "$CHUNKS" \
    --ctx-size "$CTX" \
    > "$LOG" 2>&1

# Extract metrics
PPL=$(grep "^Final estimate: PPL" "$LOG" | grep -oP 'PPL = \K[0-9.]+' || echo "NaN")
TOKPS=$(grep "prompt eval time" "$LOG" | grep -oP '[0-9.]+(?= tokens per second)' || echo "0")
RSS_KB=$(grep "Maximum resident set size" "$LOG" | grep -oP '[0-9]+' || echo "0")
RSS_MB=$(echo "scale=1; $RSS_KB / 1024" | bc 2>/dev/null || echo "0")

# Sanity check
if [[ "$PPL" == "NaN" ]] || [[ "$PPL" == "inf" ]] || [[ "$PPL" == "Inf" ]] || [[ -z "$PPL" ]]; then
    echo "SANITY_FAIL: PPL is $PPL"
    echo "---"
    echo "ppl:        $PPL"
    echo "tok_per_s:  $TOKPS"
    echo "rss_mb:     $RSS_MB"
    exit 1
fi

echo "---"
echo "ppl:        $PPL"
echo "tok_per_s:  $TOKPS"
echo "rss_mb:     $RSS_MB"
exit 0
