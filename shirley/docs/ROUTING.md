# Routing in Shirley

## The Principle

In conventional neural networks, data moves through a fixed sequence of learned transformations. In Shirley, the transformations are frozen and routing moves them to the data. Routing is the only degree of freedom.

> "Don't learn what you can read." — TriX

## Two Dimensions of Routing

### 1. Semantic Routing: Which shape?

Given an input, which frozen shape should process it? This is the TriX contribution — routing derived from the structure of the shapes themselves, not from a learned gating network.

```
signature = shape_weights.sum(dim=0).sign()   # What the shape wants
score     = input @ signature                  # How well input matches
route     = (score == scores.max())            # Send to best match
```

- Signatures are ternary vectors extracted from weight structure
- Matching uses Hamming distance (XOR + POPCNT — hardware-native on both CPU and iGPU)
- No learned router parameters — routing is zero-parameter
- Routing cost is O(active shapes), not O(all shapes), due to exponential sparsity from signature matching

### 2. Substrate Routing: Which silicon?

Given a shape, which hardware substrate executes it? This is Shirley's contribution — routing derived from the prime composition of the shape and the hardware topology.

**Rule:** A shape's substrate is determined by its prime dependency graph.

| Shape contains | Substrate | Reason |
|---------------|-----------|--------|
| Only ADD, MUL, MAX, CONST | CPU | All primes native, single-cycle, SIMD-wide |
| EXP and/or LOG | iGPU (transcendental components) | Hardware transcendental units, 4x faster than software |
| Mixed | Split dispatch | CPU-native primes on CPU, transcendentals on iGPU, synchronized via shared memory |

Substrate routing is **deterministic and static**. It's computed once when a shape is compiled from its prime composition. It never changes at runtime. There is no scheduling decision — the shape's structure *is* the schedule.

### The two dimensions are orthogonal

Semantic routing (which shape?) changes per input. Substrate routing (which silicon?) is fixed per shape. They compose without interference:

```
input arrives
  → semantic routing selects shape (dynamic, per-input)
    → substrate routing dispatches to hardware (static, per-shape)
      → computation executes on native silicon
        → result returns via shared memory
```

## Routing as the Experiment Surface

In the LEMM autoresearch loop applied to Shirley:

| Component | Maps to |
|-----------|---------|
| Experiment surface | Routing logic — signature generation, matching thresholds, composition rules, dispatch strategy |
| Evaluation harness | Frozen shapes + six primes + fixed benchmark task + accuracy metric |
| What the agent modifies | How inputs find their shapes, how shapes compose, how substrate dispatch works |
| What the agent cannot modify | The six primes, the shape implementations, the evaluation metric |

The agent evolves routing. Everything else is read-only.

## Routing Properties

### Zero-parameter
Semantic routing has no learned parameters. Signatures are derived from weight structure. Substrate routing has no parameters at all — it's determined by prime composition.

### Interpretable
Every routing decision is traceable. Input X went to shape Y because signature match score was Z. Shape Y executed on substrate W because it contains prime P. There is no black box in the routing path.

### Composable
Shapes can route to other shapes. A high-level shape (e.g., "attention head") can be a routing node that dispatches sub-computations to lower-level shapes (e.g., "softmax" routes its EXP components to iGPU). Routing is hierarchical.

### Hardware-honest
The routing doesn't pretend the hardware is uniform. It reads the topology — which primes are native where — and dispatches accordingly. A shape that's fast on one substrate and slow on another is always sent to the fast one. The routing *respects* the silicon rather than abstracting over it.

## Open Questions

1. **Dispatch granularity:** Should substrate routing dispatch entire shapes, or individual prime operations within a shape? Coarse dispatch (whole shapes) is simpler but may leave one substrate idle while the other works. Fine dispatch (per-prime) maximizes utilization but adds synchronization overhead.

2. **Shared memory bandwidth:** CPU and iGPU share memory. When both substrates are active, memory bandwidth is contested. How does routing account for bandwidth constraints? Does the optimal dispatch strategy change under memory pressure?

3. **Shape compilation:** How are shapes compiled from prime compositions? Is there a canonical form? Can two different compositions of the same function be compared for substrate efficiency?

4. **Routing evolution:** When an evolutionary search (EntroMorph) evolves routing strategies, what does it discover? Does it find routing patterns that a human designer wouldn't — the way the Gluttony Penalty produced zero-latency prediction in trixVII?

5. **Sparsity and activation:** TriX achieves 75% sparsity via routing (only 1 of 4 tiles activated per input). In Shirley, sparsity has two dimensions — semantic sparsity (most shapes not activated) and substrate sparsity (one substrate may be idle for CPU-only shapes). How do these interact?
