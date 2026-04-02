# MTFP21 as Geometric Space

## The Insight

Numbers are positions, not quantities.

In MTFP21, value = mantissa × 3^exponent. The exponent is not a scaling factor — it is a **coordinate**. It specifies where this value lives on the base-3 geometric scale. The mantissa is the local detail at that location.

This means every MTFP21 value carries two things:
- **Where it is** (exponent — position in the geometric space)
- **What it is** (mantissa — the value at that position)

## Operations as Geometry

**Multiply** adds exponents. In geometric terms: translation. Multiplying a value by 3^k moves it k positions in the space. The mantissa combines (int32 multiply), the position shifts (exponent addition). Multiplication is movement.

**Add** requires exponent alignment. You can only add values at the same position. The alignment step — shifting one value's mantissa to match the other's exponent — is the geometric operation of bringing two points to the same location before combining them. Addition is co-location.

**The computational graph is not a sequence of operations. It is a map of positions and the geometric relationships between them.** Edges are not data flow — they are translations (multiply) and co-locations (add). The graph structure is encoded in the exponent relationships between values, not in a separate node-and-edge data structure.

## The KV Cache as Geometric Space

The KV cache is not a memory buffer. It is a set of **landmarks** in the geometric space.

Each cached K vector is a position the model has visited — a landmark. Each cached V vector is what the model observed at that landmark. When a new token arrives:

1. **Q is "where am I now"** — a new position in the space
2. **Q · K computes geometric proximity** — how close the new position is to each landmark
3. **Softmax converts distances to navigation weights** — which landmarks matter from here
4. **V interpolation is the navigation result** — what the model should see from this new position, reconstructed from what it saw at nearby landmarks

The KV cache is the model's accumulated map of the sequence. Attention is navigation through that map. New tokens add new landmarks. The geometry grows.

In MTFP21, this structure is explicit. Each cached K entry has elements with exponents — those exponents ARE the position coordinates. The dot product between Q and K isn't computing a number — it's measuring geometric alignment between two positions in the space.

## Why Float32 Hides This

IEEE 754 float32 has an exponent too. But it's an implementation detail — a trick for efficient hardware representation, hidden behind the format's bit layout. The programmer thinks in terms of "the number 3.14" not "mantissa 1.57 at position 2^1."

MTFP21 makes the exponent a first-class coordinate in base-3 space. The programmer (and the hardware) works with positions explicitly. The mantissa is the local value. The exponent is the location. Both are visible, both are meaningful, both are manipulated directly.

This is not a philosophical distinction. It changes how you implement the computation:

- **Float32 graph:** Build a data structure of nodes and edges. Allocate buffers. Route data through the graph. The structure is external to the values.
- **MTFP21 geometric space:** Values carry their own positions. Operations are geometric transformations. The structure is intrinsic to the values. No external graph needed.

## Implications for Shirley

### The ggml graph is unnecessary for MTFP21 sections

The ggml computational graph is a node-and-edge data structure that tracks what connects to what. In the MTFP21 domain, this is redundant — the values encode their own relationships through their exponents. The Shirley compute functions (shirley_ffn_compute, shirley_attn_compute) operate on the geometric space directly, not through a graph.

The ggml graph remains useful at the boundaries — float32 input from embeddings, float32 output to the LM head. But within the MTFP21 domain, the geometry IS the graph.

### The KV cache is an MTFP21 geometric structure

Each cached entry is a (mantissa, exponent) pair — a position in the geometric space. The cache stores these directly, in the native MTFP21 format. No conversion to float16. No separate indexing structure. The exponents encode the spatial relationships.

Sparsity is native: a zero mantissa means "this position doesn't matter." Cache entries can be pruned by zeroing mantissa trits without rebuilding any data structure. The geometry adapts.

### Attention is geometric navigation

Q@K^T is not "matrix multiplication" — it is measuring geometric proximity between a new position (Q) and all landmarks (K cache). Softmax is not "normalizing scores" — it is converting geometric distances to navigation weights. V interpolation is not "weighted sum" — it is reconstructing the view from a new position using observations at nearby landmarks.

These operations are the six primes applied to the geometric space:
- **MUL:** Translation between positions (Q · K dot product)
- **ADD:** Co-location for accumulation (weighted V sum)
- **MAX:** Selecting the dominant landmark (softmax stabilization)
- **EXP:** Converting geometric distance to weight (softmax)
- **CONST:** The precomputed landmarks (cached K/V positions, RoPE sin/cos tables)
- **LOG:** Not used in forward pass — available for information-theoretic operations

### Routing in geometric space

The TriX routing principle — "don't learn what you can read" — acquires new meaning in the geometric framing. Routing is not selecting which function to apply. It is navigating the geometric space to find the right position. The signature matching (input @ signature) is measuring geometric proximity. The routing decision is: which region of the space does this input belong in?

In the KV cache: the attention mechanism IS routing. It reads the geometric space (K cache) to find which landmarks the input is close to, then navigates to the appropriate region (V interpolation). Attention and routing are the same operation in different parts of the architecture.

## Origin

Insight by A.N. Josserand-Austin, April 1, 2026. "Zoom out and think of numbers as positions instead of quantities."
