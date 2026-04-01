#!/bin/bash
# test.sh — Research-grade inference test with full parameter logging
#
# Usage: ./test.sh <baseline|shirley> "prompt text" [options]
#
# All hyperparameters, inputs, outputs, and timing are logged to
# builds/logs/<variant>_<timestamp>.log
#
# Options (override defaults):
#   --tokens N    max tokens (default: 512)
#   --seed N      RNG seed (default: 42)
#   --temp F      temperature (default: 0.7)
#   --threads N   thread count (default: 6)
#   --ctx N       context size (default: 4096)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="$SCRIPT_DIR/../models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

# Parse arguments
VARIANT="${1:-}"
PROMPT="${2:-}"

if [[ -z "$VARIANT" || -z "$PROMPT" ]]; then
    echo "Usage: $0 <baseline|shirley> \"prompt text\" [--tokens N] [--seed N] [--temp F] [--threads N]"
    exit 1
fi

shift 2

# Defaults
TOKENS=512
SEED=42
TEMP=0.7
THREADS=6
CTX=4096

# Parse optional overrides
while [[ $# -gt 0 ]]; do
    case "$1" in
        --tokens)  TOKENS="$2"; shift 2 ;;
        --seed)    SEED="$2"; shift 2 ;;
        --temp)    TEMP="$2"; shift 2 ;;
        --threads) THREADS="$2"; shift 2 ;;
        --ctx)     CTX="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Select binary
BIN_DIR="$SCRIPT_DIR/$VARIANT"
if [[ ! -f "$BIN_DIR/run.sh" ]]; then
    echo "Error: variant '$VARIANT' not found at $BIN_DIR"
    exit 1
fi

# Timestamp for log file
TS=$(date +%Y%m%d_%H%M%S)
LOGFILE="$LOG_DIR/${VARIANT}_${TS}.log"

# Record everything
{
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  SHIRLEY INFERENCE TEST LOG                                 ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo ""
    echo "=== METADATA ==="
    echo "timestamp:   $(date -Iseconds)"
    echo "variant:     $VARIANT"
    echo "host:        $(hostname)"
    echo "cpu:         $(lscpu | grep 'Model name' | sed 's/.*:\s*//')"
    echo "git_commit:  $(cd $SCRIPT_DIR/.. && git rev-parse --short HEAD)"
    echo "git_branch:  $(cd $SCRIPT_DIR/.. && git branch --show-current)"
    echo ""
    echo "=== HYPERPARAMETERS ==="
    echo "model:       $MODEL"
    echo "model_hash:  $(md5sum "$MODEL" | cut -d' ' -f1)"
    echo "tokens:      $TOKENS"
    echo "seed:        $SEED"
    echo "temperature: $TEMP"
    echo "threads:     $THREADS"
    echo "ctx_size:    $CTX"
    echo "binary_hash: $(md5sum "$BIN_DIR/llama-cli" | cut -d' ' -f1)"
    echo "libggml_hash: $(md5sum "$BIN_DIR/libggml.so" | cut -d' ' -f1)"
    echo ""
    echo "=== INPUT ==="
    echo "prompt: |"
    echo "  $PROMPT"
    echo ""
    echo "=== COMMAND ==="
    echo "$BIN_DIR/run.sh -m $MODEL -p \"$PROMPT\" -n $TOKENS -t $THREADS --seed $SEED --temp $TEMP --ctx-size $CTX"
    echo ""
    echo "=== OUTPUT ==="
} > "$LOGFILE"

# Run inference, capture everything
"$BIN_DIR/run.sh" \
    -m "$MODEL" \
    -p "$PROMPT" \
    -n "$TOKENS" \
    -t "$THREADS" \
    --seed "$SEED" \
    --temp "$TEMP" \
    --ctx-size "$CTX" \
    2>&1 | tee -a "$LOGFILE"

{
    echo ""
    echo "=== END ==="
    echo "log_file:    $LOGFILE"
} >> "$LOGFILE"

echo ""
echo "Log saved: $LOGFILE"
