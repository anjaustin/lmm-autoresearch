# Applying LEMM to Shirley

## The Setup

The Lincoln-Einstein Manifold Method (LEMM) — the hardened autoresearch protocol with micro-manifold reasoning, journal, arc archiving, and red-teaming — applied to discovering optimal routing strategies for frozen shape computation on heterogeneous silicon.

### Component Mapping

| LEMM Component | Shirley Implementation |
|----------------|----------------------|
| **Experiment surface** | Routing code — signature generation, matching logic, composition rules, substrate dispatch |
| **Evaluation harness** | Six primes (frozen) + shape library (frozen) + benchmark task + accuracy/throughput metric |
| **Program (program.md)** | Domain knowledge about primes, shapes, hardware topology, routing constraints |
| **Journal** | Reasoning about routing strategies, failure patterns in the search space, hardware observations |
| **Strategy override** | Human steering — "explore semantic routing variations" or "focus on dispatch granularity" |
| **results.tsv** | Experiment log with hypotheses and outcomes |

### What the Agent Can Modify
- How signatures are generated from shape structure
- How input-to-shape matching works (thresholds, distance metrics, tiebreaking)
- How shapes compose and nest
- How substrate dispatch decisions are made for mixed-prime shapes
- The routing topology — which shapes are available, how they're organized

### What the Agent Cannot Modify
- The six primes (ADD, MUL, EXP, LOG, MAX, CONST)
- The implementation of individual primes on each substrate
- The evaluation metric and benchmark data
- The hardware topology (CPU capabilities, iGPU capabilities, shared memory)

## The Metric

Two candidates, depending on what we're optimizing:

### Option A: Accuracy at fixed compute budget
- Run the benchmark task with a fixed wall-clock budget
- Measure task accuracy (e.g., classification accuracy, function approximation error)
- The agent optimizes routing to maximize accuracy within the budget
- Analogous to Karpathy's original: best model in 5 minutes

### Option B: Compute cost at fixed accuracy
- Define a minimum accuracy threshold
- Measure total compute (cycles, energy, latency) to reach that threshold
- The agent optimizes routing to minimize compute while meeting the accuracy bar
- More relevant to Shirley's edge deployment thesis

### Option C: Composite
- `score = accuracy / (compute_cost * complexity_penalty)`
- Single scalar, higher is better
- Balances quality against efficiency against simplicity

The choice depends on the benchmark task. For an initial proof-of-concept, Option A is simplest — it mirrors the original autoresearch protocol directly.

## The Benchmark Task

Requirements for a good initial benchmark:
- Runs on CPU + iGPU (no discrete GPU needed)
- Completes within a fixed time budget (e.g., 2-3 minutes)
- Has a clear scalar metric
- Exercises all six primes (so substrate routing matters)
- Is well-understood enough that improvements are meaningful

Candidates from existing work:
- **MNIST classification via frozen shapes** — SSTT already achieves 97.27% with zero parameters; can Shirley's routing match or exceed this?
- **Function approximation** — approximate a known function (e.g., sin, Bessel, FFT) using composed frozen shapes; measure approximation error
- **Anomaly detection** — Flynn's domain; detect anomalies in CWRU bearing data using routed frozen shapes; measure detection accuracy and false positive rate
- **Binary arithmetic** — FLYNNCONCEIVABLE's domain; compute binary operations using routed shapes; measure correctness across exhaustive input space

## The Loop

```
SESSION START:
    1. Read program.md, strategy_override.md, results.tsv
    2. Read journal for prior routing discoveries
    3. Understand current routing implementation

MICRO-MANIFOLD → EXPERIMENT → RED-TEAM (per the base protocol)
    - Hypothesis about a routing change
    - NODES: what do prior experiments show about this area of routing space?
    - REFLECT: is this change principled or random?
    - SYNTHESIZE: modify the routing code
    - Run benchmark
    - Log result
    - If kept: red-team — is this improvement genuine? Robust? Clean?

ARC ARCHIVING:
    When stuck on a routing strategy, archive the arc.
    Start fresh from baseline routing with journal understanding.
    Each arc explores a different region of routing space.
    Einstein restarts.
```

## What Makes This Different from Standard Autoresearch

1. **The experiment surface is routing logic, not model architecture.** The agent isn't tweaking hyperparameters on a neural network. It's evolving how computation finds its way to data and silicon.

2. **The frozen components are genuinely frozen.** In Karpathy's setup, the agent modifies everything in train.py — architecture, optimizer, hyperparameters. In Shirley, the primes and shapes are untouchable. The agent can *only* change routing. This is a much tighter constraint, which may produce more interesting discoveries.

3. **Hardware topology is part of the search space.** The agent isn't optimizing for an abstract metric on uniform hardware. It's discovering how to use heterogeneous silicon efficiently. Routing strategies that ignore the substrate will perform worse than those that respect it.

4. **No GPU required.** The entire loop runs on a Ryzen with an iGPU. The "GPU" in this setup is the integrated graphics unit, accessed via compute shaders or OpenCL, not CUDA. This is commodity hardware.

5. **The search may discover routing patterns that emerge from the hardware constraint.** The way the Gluttony Penalty in trixVII produced zero-latency prediction — an emergent behavior that wasn't designed, just discovered — Shirley's hardware constraint may produce routing strategies that a human wouldn't design. The constraint is the creative pressure.

## Prerequisites Before First Run

- [ ] Define the benchmark task and metric
- [ ] Implement the six primes as frozen, callable functions on both substrates
- [ ] Implement a baseline routing strategy (e.g., TriX-style signature matching)
- [ ] Build the evaluation harness (benchmark runner + metric computation)
- [ ] Set up the file structure (results.tsv, strategy_override.md, journal/)
- [ ] Write program.md with Shirley-specific domain knowledge
- [ ] Verify CPU prime execution (AVX2 kernels)
- [ ] Verify iGPU prime execution (compute shader or OpenCL for EXP/LOG)
- [ ] Run baseline, establish reference metric
