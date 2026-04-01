# lmm-autoresearch

A hardened, domain-agnostic protocol for autonomous AI-driven optimization — built on [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) and refined through six iterations of red-teaming and design.

## What changed

Karpathy's original is an elegant loop: an AI agent modifies code, trains for 5 minutes, checks if the result improved, keeps or discards, repeats. You sleep; it researches.

This fork keeps the core loop and adds a reasoning layer:

- **Micro-manifold** — a compressed [Lincoln Manifold Method](https://github.com/anjaustin/lmm) cycle before each experiment. The agent consults the record, forms a hypothesis, challenges its assumptions, *then* writes code. After each kept result, the agent red-teams its own success before building on it.
- **Journal** — persistent reasoning artifacts (`journal/`) that survive session boundaries and arc archives. The agent documents not just what happened but *why it expected something different*, and what failures reveal about the search space.
- **Research arc archiving** — when stuck, the agent archives the branch instead of nuking to baseline. Nothing is destroyed. The human decides what to prune after review.
- **Session rotation** — fixed-length sessions prevent context window degradation. Each session starts fresh; the journal carries the understanding.
- **Async strategy override** — the human can steer mid-session via `strategy_override.md` without interrupting the loop.
- **Failure as scaffolding** — failures are documented as structural findings, not waste. A pattern of failures is itself a discovery.
- **Base protocol + hardening tier** — the base protocol works with zero configuration. An optional hardening tier adds calibrated thresholds (acceptance, holdout divergence, parameter bounds) for users with domain expertise.

For the full protocol, design rationale, and hardening analysis, see [`AUTORESEARCH.md`](AUTORESEARCH.md).

## Applied: BitNet ternary conversion (Shirley)

This protocol has been applied to the [BitNet b1.58-2B-4T](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-gguf) end-to-end ternary conversion — replacing float32 operations with ternary integer operations in a 2B-parameter LLM, one operation at a time. 11 experiments across 6 phases mapped the precision requirements of every operation in the transformer pipeline.

Key finding: 4 balanced ternary trits (81 levels, 6.3 bits) is sufficient for matmul outputs and activation functions across all 28 layers. Only RMSNorm requires higher precision (7 trits). See [`shirley/docs/BITNET_TERNARY_PLAN.md`](shirley/docs/BITNET_TERNARY_PLAN.md) for the full plan and results.

## How it works

The repo has four files that matter:

- **`prepare.py`** — fixed constants, one-time data prep (downloads training data, trains a BPE tokenizer), and runtime utilities (dataloader, evaluation). Not modified.
- **`train.py`** — the single file the agent edits. Contains the full GPT model, optimizer (Muon + AdamW), and training loop. Everything is fair game. **This file is edited and iterated on by the agent.**
- **`program.md`** — agent instructions implementing the hardened protocol. Point your agent here and let it go. **This file is edited and iterated on by the human.**
- **`AUTORESEARCH.md`** — the protocol itself: philosophy, constraints, micro-manifold, journal structure, arc archiving, red-teaming, hardening tier, design rationale, and limitations.

Training runs for a **fixed 5-minute time budget** (wall clock, excluding startup/compilation). The metric is **val_bpb** (validation bits per byte) — lower is better, vocab-size-independent.

## Quick start

**Requirements:** A single NVIDIA GPU (tested on H100), Python 3.10+, [uv](https://docs.astral.sh/uv/).

```bash
# 1. Install uv project manager (if you don't already have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Download data and train tokenizer (one-time, ~2 min)
uv run prepare.py

# 4. Manually run a single training experiment (~5 min)
uv run train.py
```

If the above commands all work, your setup is ready for autonomous research.

## Running the agent

Spin up Claude Code (or your preferred agent) in this repo, then prompt:

```
Read program.md and let's kick off a new experiment. Do the setup first.
```

The agent will create a branch, initialize `results.tsv` and the journal, establish a baseline, and begin the experiment loop with the micro-manifold reasoning cycle.

## Project structure

```
prepare.py              — constants, data prep + runtime utilities (do not modify)
train.py                — model, optimizer, training loop (agent modifies this)
program.md              — agent instructions (human modifies this)
AUTORESEARCH.md         — protocol document: philosophy, design, hardening
pyproject.toml          — dependencies
journal/                — session reasoning artifacts (gitignored, created at runtime)
strategy_override.md    — async human steering file (gitignored, created at runtime)
results.tsv             — experiment log (gitignored, created at runtime)
```

## Design philosophy

The protocol went through six iterations — from Karpathy's elegant original, through an over-hardened version with 13 steps and 10+ parameters, back down to a triaged design, then refined with the Lincoln Manifold Method, and finally enriched with a reasoning layer and non-destructive arc management. The full design history is in [`AUTORESEARCH.md`](AUTORESEARCH.md).

The guiding principle: **if the guardrails need guardrails, cut the guardrails.** The micro-manifold and journal aren't guardrails — they're cognition. They don't constrain the agent's behavior; they improve the agent's thinking. Arc archiving isn't a guardrail — it's preservation. The nuke is still available; it's just a human decision, not an agent decision under pressure.

## Platform support

This code requires a single NVIDIA GPU. See the [upstream repo](https://github.com/karpathy/autoresearch) for platform-specific forks (MacOS/MLX, Windows/RTX, AMD/ROCm) and guidance on adapting for smaller compute.

## Upstream

Forked from [karpathy/autoresearch](https://github.com/karpathy/autoresearch). The training code (`prepare.py`, `train.py`) is unchanged. The protocol layer (`program.md`, `AUTORESEARCH.md`) is new.

## Credits

- Original autoresearch design by [Andrej Karpathy](https://github.com/karpathy) (March 2026)
- Protocol analysis, red-teaming, and hardened design by A.N. Josserand-Austin and Claude (March 2026)
- Reasoning layer adapted from the [Lincoln Manifold Method](https://github.com/anjaustin/lmm) by A.N. Justin

## License

MIT
