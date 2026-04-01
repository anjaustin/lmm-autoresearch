# The Autoresearch Loop

A domain-agnostic protocol for autonomous AI-driven optimization. The method is not specific to LLM training — it works for any problem expressible as: *modify code, run, measure, keep or discard*.

This document describes the hardened protocol — the original design, red-teamed, triaged, and refined through six iterations. The base protocol works with zero configuration. An optional hardening tier adds calibrated guardrails for users with domain expertise and evaluation infrastructure.

## The Core Abstraction

Three components, strictly separated:

| Component | Role | Who edits it |
|-----------|------|--------------|
| **Experiment surface** | The code being optimized (e.g. `train.py`) | Agent only |
| **Evaluation harness** | Fixed ground-truth measurement + sanity checks (e.g. `prepare.py`) | Nobody — read-only |
| **Program** | Instructions, strategy, constraints (e.g. `program.md`) | Human only |

Three auxiliary structures support the loop:

| Structure | Role | Who edits it |
|-----------|------|--------------|
| **`results.tsv`** | Experiment log — every run, every outcome, every hypothesis | Agent appends |
| **`strategy_override.md`** | Live steering — human can redirect mid-session | Human only, async |
| **`journal/`** | Structured reasoning artifacts — one file per session | Agent writes during session |

The separation is load-bearing. The agent can't game the metric because it can't touch the harness. The human can steer without stopping the loop. The experiment surface is the only degree of freedom. The journal preserves understanding across resets and sessions.

## results.tsv Schema

```
commit	primary	holdout	status	hypothesis	description
```

**Status values:**

| Status | Meaning |
|--------|---------|
| `keep` | Metric improved, commit advanced |
| `discard` | Metric did not improve, reverted |
| `crash` | Experiment crashed |
| `provisional` | Metric regressed but kept for one more experiment |
| `recovered` | Follow-up to provisional that recovered (both kept) |
| `overfit` | Holdout divergence triggered revert (hardening tier) |
| `session_summary` | End-of-session structured summary |
| `override_ack` | Agent acknowledges a strategy override |
| `arc_archived` | Research arc archived, new arc started from baseline |

The `holdout` column is empty unless the hardening tier is enabled.

## The Micro-Manifold

Before each experiment, the agent runs a compressed reasoning cycle adapted from the Lincoln Manifold Method. This is structured thinking, not structured documentation — the agent writes to the session journal, not to separate files per phase.

```
PRE-EXPERIMENT:
  RAW     →  What's my idea? What do I expect? (becomes the hypothesis column)
  NODES   →  What do prior experiments in this area show? What tensions exist?
  REFLECT →  Am I trying this because I have a reason, or because I ran out of ideas?
              What assumption am I making that might be wrong?
  SYNTHESIZE → The actual code change, informed by the first three steps.

POST-EXPERIMENT (on keep):
  RED-TEAM →  What could be wrong with this result?
              Is the improvement genuine, or could it be noise, overfitting,
              or an artifact? Did I introduce complexity, coupling, or a
              shortcut that will constrain future work?
```

The pre-experiment cycle (RAW → SYNTHESIZE) is the hypothesis. The post-experiment red-team is the scrutiny. Both are written to the session journal.

The RAW phase is one line in results.tsv. NODES and REFLECT are 2-5 lines each in the journal. The RED-TEAM phase is 2-5 lines in the journal, written only on `keep` results — when the agent is about to advance the branch and commit to building on top of this change.

**Why red-team on keep:** A `discard` is self-correcting — the code reverts, no harm done. A `keep` is permanent — every future experiment builds on it. Accepting an improvement without scrutiny means building on an unexamined foundation. If three flagged improvements stack up, the branch may be building on sand. The red-team catches this before it compounds.

If the red-team identifies a serious concern, the agent logs it as a **flag** in the journal. The improvement still stands — the metric improved, and the protocol respects that. But the doubt travels with it. Flags are visible to the human during review and to the agent in future sessions. An accumulation of flags is a signal to pause and examine the foundation.

**Why this matters:** The protocol's deepest limitation was always the search/research gap. The agent mutated code and checked a number. The pre-experiment manifold forces the agent to understand before it acts. The post-experiment red-team forces the agent to *question its own successes* before building on them. Together, they make the search deliberate in both directions — skeptical of ideas before trying them, skeptical of results after getting them.

## The Journal

### Structure

```
journal/
├── session_001.md
├── session_002.md
├── session_003.md
└── ...
```

One file per session. Each file contains micro-manifold entries — the NODES, REFLECT, and RED-TEAM phases for each experiment in that session, plus a session-end reflection.

### What goes in a journal entry

```markdown
## Experiment 14: [one-line description]

**Nodes:** Prior experiments show that LR changes above 0.06 consistently
crashed (exps 3, 7, 11). Optimizer momentum was the strongest lever in
session 1. Tension: higher LR improves early convergence but causes
late-training instability.

**Reflect:** I'm trying a LR schedule that decays faster because the
instability pattern suggests the model needs lower LR in the second half,
not a lower peak. This is different from exp 7 which just lowered peak LR.
Assumption I might be wrong about: the instability might be batch-size
related, not LR related.

**Result:** Kept. Primary improved 0.9871 → 0.9842. Hypothesis confirmed —
faster decay stabilized late training without hurting early convergence.
This validates the pattern from exps 3/7/11: the bottleneck is late-stage
stability, not peak LR.

**Red-team:** The improvement is real but the decay schedule is now tightly
coupled to the current model depth and batch size. If either changes in a
future experiment, this LR schedule may become a hidden constraint.
Also: 0.0029 BPB gain is above noise floor but not by much — worth
watching whether the next few experiments hold or erode this gain.
⚑ FLAG: LR schedule may be fragile to architecture changes.
```

Not an essay. Not a formality. Structured thinking, externalized.

### Red-teaming successes

Every `keep` gets a red-team entry in the journal before the agent moves on. The agent asks:

- **Is this improvement genuine?** Could it be stochastic noise? Does the magnitude exceed the observed variance of prior runs?
- **Is this improvement robust?** Did I introduce a coupling, a shortcut, or a hidden dependency that will constrain future experiments?
- **Is this improvement clean?** Did the code get simpler or more complex? Is the complexity justified by the gain?
- **What would break this?** If I change the architecture, the optimizer, or the batch size next, does this improvement survive?

If the red-team identifies a concern, the agent marks it with `⚑ FLAG:` in the journal. Flags don't trigger reverts — the metric improved, and the protocol respects that. But flags travel with the improvement. They're visible to the agent in future micro-manifold NODES phases and to the human during review.

**Accumulation of flags is a signal.** If three or more consecutive kept experiments carry flags, the agent should note this pattern in its REFLECT phase and consider whether the branch is building on a fragile foundation. The human reviewing the journal can see the flag density and decide whether to intervene via strategy override.

### Documenting failures

Failures are not just entries to skip past — they are structural scaffolding that reveals the shape of the search space. A discarded experiment is load-bearing information.

When an experiment fails, the journal entry should capture not just *what* failed but *what the failure tells you*:

```markdown
## Experiment 15: GeLU activation swap

**Nodes:** ReLU-squared has been the activation since baseline. Three prior
attempts to change activation (exps 5, 9, 12) were all reverted.

**Reflect:** Trying GeLU because it's smoother than ReLU-squared and might
help the late-training stability issue. But the pattern of three failed
activation swaps suggests the model architecture may be tightly coupled
to the activation shape — not just preference, but structural dependence.

**Result:** Discarded. Primary regressed 0.9842 → 0.9910. Fourth failed
activation change. This is no longer "wrong activation" — this is
evidence that the current architecture *depends* on ReLU-squared's
sparsity pattern. Future experiments should treat the activation as
fixed and work around it, or change the architecture simultaneously.
```

The failure itself is the finding. Four failed activation swaps is a discovery: the architecture and activation are coupled. That insight redirects future search more powerfully than any single successful experiment.

Over time, the journal accumulates a map of the search space — not just where the peaks are, but where the walls are, and *why* they're walls. This map survives arc archives and session boundaries. It is the agent's most valuable output after the code itself.

### Session-end reflection

At the end of each session, the agent writes a brief reflection in the journal:

```markdown
## Session 3 Reflection

**What worked:** Decaying LR schedule (exp 14) was the biggest single gain.
Momentum changes continue to be reliable.

**What didn't:** Every activation function change was reverted (exps 5, 9,
12, 15). This isn't "wrong activations" — it's evidence that the
architecture is structurally coupled to ReLU-squared's sparsity pattern.
Treat activation as fixed unless changing architecture simultaneously.

**Emerging pattern:** The model is bottlenecked by late-training stability,
not early convergence speed. Next session should focus on warmdown schedules
and gradient clipping.

**Current best:** commit a1b2c3d, primary 0.9842
```

This is written *during* the session as part of the thinking process — it's not a retrospective report card. It's the agent's real-time understanding, captured before the session ends and context is lost.

### Research arcs and archiving

When the agent gets stuck — thrashing against a local optimum with no progress — it doesn't nuke the branch. It *archives* it.

```
ARCHIVE ARC:
    1. Tag or rename the current branch (e.g. autoresearch/mar5-arc1)
    2. Log an arc_archived row to results.tsv with the archived branch name,
       best metric achieved, and reason for archiving
    3. Create a new branch from baseline (e.g. autoresearch/mar5-arc2)
    4. Read the journal for accumulated understanding
    5. Begin the new arc with informed hypotheses, not random mutations
```

The archived branch preserves the code — every commit, every improvement, the full implementation history. The journal preserves the understanding. The new arc starts from baseline with both available.

**Why archive instead of nuke:** A baseline reset destroyed the code and forced the agent to re-derive implementations from conceptual knowledge in the journal. That was expensive — the journal tells you *what* worked, but re-implementing it costs time and may not reproduce exactly. With archiving, the prior arc's code is one `git diff` or `git show` away. The agent can read the archived branch to see *how* something was built, not just *that* it was built.

**The human decides what dies.** After reviewing the research arcs — the code, the metrics, the journal — the human decides whether to delete archived branches, cherry-pick specific commits into the active arc, merge arcs, or keep the archives for reference. The nuke is still available; it's just a human decision at review time, not an agent decision under pressure.

### Journal hygiene

- The `journal/` directory is `.gitignore`d — it's research scaffolding, not committed output
- Journal entries are written during the session, not retroactively
- The journal is additive — the agent never edits prior session files
- If the journal grows large (many sessions), the agent reads the most recent 2-3 session files in full and scans session-end reflections from older files

## The Loop (Base Protocol)

```
SESSION START:
    1. Read program.md, strategy_override.md, and results.tsv
    2. Check git state — current branch, current best commit
    3. Read session_summary rows for compressed history
    4. Read recent journal files (last 2-3 sessions)
    5. Create journal/session_NNN.md for this session
    6. If starting a new arc after archiving:
       Read journal and optionally inspect archived branch for reference

LOOP UNTIL SESSION LIMIT:
    1. Check strategy_override.md for new human instructions
       If changed → log an override_ack row with interpretation
    2. MICRO-MANIFOLD:
       a. RAW: Form hypothesis (one line — what you expect and why)
       b. NODES: Consult results.tsv and journal — what does the record show
          about this area of the search space? What tensions exist?
          Write 2-5 lines to session journal.
       c. REFLECT: Why this idea? What assumption might be wrong?
          Write 2-5 lines to session journal.
       d. SYNTHESIZE: Make the code change, informed by the above.
    3. Commit the change
    4. Run the experiment under fixed time budget
    5. Harness returns:
       a. Primary metric (reported to agent)
       b. Sanity pass/fail (NaN/Inf check)
    6. Accept/reject:
       - Sanity failure → revert, log, move on (non-negotiable)
       - Primary improved → keep
       - Primary not improved + no provisional outstanding → revert
       - Primary not improved + provisional available → mark provisional
         (one provisional max, no chaining)
       - Next experiment after provisional:
           Recovered → keep both, log as "recovered"
           Not recovered → revert both
    7. Log to results.tsv
    8. Update journal with result — one line: what actually happened
       vs. hypothesis. Note surprises.
    9. If kept → RED-TEAM the result:
       Write 2-5 lines to journal. Is this genuine? Robust? Clean?
       What would break it? Flag concerns with ⚑ FLAG.
       (see "Red-teaming successes")
   10. Thrash check: if N consecutive reverts without progress,
       archive the current arc and start a new one from baseline
       (see "Research arcs and archiving")

SESSION END:
    1. Write session-end reflection to journal
    2. Append a session_summary row to results.tsv:
       - Current best commit and primary metric
       - Session stats: keeps / discards / crashes
       - Current LOC of the experiment surface
    The agent stops. The next session reads results.tsv, the journal,
    and the branch.
```

## Base Protocol Features

These require no configuration and no additional infrastructure beyond the original Karpathy setup.

### Session rotation
The agent does not run in one infinite session. Sessions have a fixed experiment limit (e.g. 20 experiments, ~2 hours). At session end, the agent writes a session-end reflection to the journal, appends a summary row to results.tsv, and stops. A fresh session picks up from the branch state, the log, and the journal.

This prevents context window degradation — the failure mode where the agent gets dumber the longer it runs. Each session starts with full context capacity. The journal carries the understanding; the log carries the data; the fresh context carries the capacity.

### Strategy override with acknowledgment
The human can steer mid-session without interrupting the loop. `strategy_override.md` is checked at the start of each loop iteration. If modified, the agent reads the new instructions and logs an `override_ack` row to results.tsv with a brief statement of how it's interpreting the override.

This gives the human both a steering mechanism and visibility into whether the agent understood the instruction. The feedback cycle drops from once-per-day to minutes.

### Failure memory
Before proposing an experiment, the agent reads `results.tsv` and the journal to check whether a similar idea has already been tried and discarded. If so, it either tries something else or explains what's different. The journal provides richer context than the TSV alone — not just *what* was tried but *why it was expected to work*, *why it didn't*, and *what the failure revealed about the search space*.

Failures are not obstacles to route around. They are structural findings. A pattern of failures in a region of the search space is itself a discovery — evidence of a constraint, a coupling, or a wall. The agent should treat accumulated failures as load-bearing scaffolding for future hypotheses, not just a list of things to avoid.

### Hypothesis tracking
Before each experiment, the agent writes a one-line hypothesis in the `hypothesis` column: what it expects to happen and why. After the experiment, the journal entry captures the comparison: what actually happened vs. what was expected.

Over time, this creates a record of the agent's predictive accuracy. Consistently wrong hypotheses signal that the agent is guessing rather than reasoning — visible to both the agent (in the journal) and the human (in the TSV).

### Single provisional commit
The agent can keep one experiment that didn't improve the metric, marked `provisional`, on the bet that the next experiment will recover. If the next experiment recovers (metric improves past the pre-provisional best), both are kept. If not, both revert.

No chaining — at most one provisional commit outstanding at any time.

### NaN/Inf sanity check
The harness checks for NaN/Inf in model weights after each experiment. Failure triggers automatic revert regardless of metric. The agent cannot override.

### Session summary rows
At session end, the agent appends a row with status `session_summary`. The description contains verifiable facts: current best commit hash and metric, session keep/discard/crash counts, current LOC of the experiment surface.

### LOC tracking
The session summary row includes the current line count of the experiment surface. The human sees growth over time. No automated gate — just visibility.

## The Constraints

### Fixed time budget
Every experiment runs for exactly the same wall-clock duration. Makes all results directly comparable regardless of what the agent changes. The agent optimizes *for your hardware*.

**Acknowledged bias:** You're optimizing for "best in T minutes on this GPU," not "best model." Architectures needing longer warmup are excluded. The human can manually test the best result at longer budgets to check for sprinter bias.

### Single scalar metric
One number, one direction (lower or higher). No ambiguity. Must be config-independent so architectural changes are fairly compared.

### Git-based keep/revert with arc archiving
Every experiment is a commit. Good results advance the branch. Bad results revert. On thrash (many consecutive reverts without progress), the agent archives the current research arc and starts a new branch from baseline.

The archived arc preserves every commit. The journal preserves the reasoning. The agent reads both before re-approaching the problem on the new arc. Nothing is destroyed — the human decides what to delete, merge, or cherry-pick after reviewing the arcs.

### Simplicity criterion
All else equal, simpler is better. This is a judgment call by the agent, verified by human review of diffs. The protocol does not try to formalize it — LOC gates and complexity scores measure syntax, not semantics. The LOC tracking in session summaries gives the human visibility; enforcement is human.

## Hardening Tier (Optional)

These features require calibration and/or additional evaluation infrastructure. Enable them when you have domain expertise and the data to set thresholds correctly. The base protocol works without them.

### Acceptance threshold

An improvement must exceed a minimum delta to count. Below the threshold, the result is treated as noise and reverted.

**How to calibrate:** Run the baseline experiment 3-5 times without changes. Measure the variance in your primary metric. Set the acceptance threshold above the observed variance (e.g. 2x the standard deviation). If you can't run repeated baselines, start without this feature and add it after you've seen enough results to estimate variance from the log.

**Risk of miscalibration:** Too high rejects real improvements and causes thrashing. Too low accepts noise. When in doubt, don't enable this — the base protocol still works, just with weaker noise resistance.

### Holdout metric with automated divergence revert

The harness evaluates on both a primary set and a holdout set. Both metrics are logged to the `holdout` column in results.tsv. If the holdout diverges from the primary by more than a configured threshold, the experiment is automatically reverted with status `overfit`.

**How to calibrate:** The holdout set should be drawn from the same distribution as the primary but must be a distinct, non-overlapping sample. The divergence threshold should be set based on expected natural variance between the two sets — again, run the baseline on both to establish this. Start with a generous threshold and tighten as you accumulate data.

**Risk of miscalibration:** Too tight rejects improvements that happen to affect the two sets differently. Too loose doesn't catch overfitting until it's advanced. The human should periodically review the holdout column in results.tsv regardless of whether auto-revert has triggered.

### Parameter count bounds

The harness checks that the model's parameter count stays within a configured range of the baseline (e.g. 0.5x to 2x). Outside the range triggers automatic revert.

**How to set:** Define the range based on your hardware constraints and research goals. This catches degenerate architectures (collapsed to near-zero or exploded to OOM territory).

### Thrash limit with automated arc archiving

Configure the number of consecutive reverts (N) before the agent automatically archives the current arc and starts fresh. This automates what the base protocol leaves to agent judgment.

**How to set:** A reasonable starting value is 5-8 consecutive reverts. Lower values archive more aggressively (less time wasted in dead ends, more arcs to review). Higher values give the agent more chances to find a way out (more time spent, fewer arcs).

## Multi-Agent Composition

Multiple agents can run on separate branches. Composition is a human decision. Three approaches, with tradeoffs:

**Pick the winner.** Evaluate the best commit from each branch on the same data. Keep the best. Simple, monotonic, but can't combine innovations.

**Attempt to combine.** The human (or a dedicated merge-agent) creates a synthetic commit incorporating innovations from multiple branches and runs it as a new experiment. Can discover synergies, but the combination may conflict (e.g. different activations tuned with different learning rates). The synthetic commit is just another experiment — if it doesn't beat the current best, revert.

**Split and specialize.** Assign different experiment categories to different agents (e.g. one agent explores optimizers, another explores architectures). Reduces redundant exploration but requires the human to partition the search space well. Good when the human has strong priors about which directions are independent.

None of these are mandated by the protocol. The right approach depends on the problem and the human's insight.

## Adapting to Other Domains

To apply autoresearch to a different problem:

1. **Define the experiment surface** — a single file the agent can modify. Small enough that diffs are reviewable.

2. **Build the evaluation harness** — returns a primary scalar metric and a sanity pass/fail (at minimum, NaN/Inf check). The agent cannot modify it. Pin test data, fix seeds, control the environment.

3. **Choose a time budget** — long enough for signal, short enough for iteration.

4. **Write `program.md`** — domain knowledge, constraints, strategy, session length.

5. **Create the initial structure:**
   - `results.tsv` — header row: `commit | primary | holdout | status | hypothesis | description`
   - `strategy_override.md` — initially empty
   - `journal/` — empty directory

6. **Set up git** — create a branch, commit the baseline, run. Add `journal/`, `results.tsv`, and `strategy_override.md` to `.gitignore`.

7. **(Optional) Configure hardening thresholds** — if you have the domain expertise and evaluation infrastructure, enable acceptance threshold, holdout metric, parameter bounds, and/or thrash limit as described in the Hardening Tier section.

### Example domains

| Domain | Surface | Primary | Holdout | Budget |
|--------|---------|---------|---------|--------|
| LLM pretraining | Model + optimizer | val BPB (shard A) | val BPB (shard B) | 5 min |
| Compiler passes | Pass config | Benchmark A runtime | Benchmark B runtime | 2 min |
| Game AI | Policy/heuristics | Win rate vs. baseline | Win rate vs. stronger AI | 3 min |
| Shader optimization | Shader code | Frame time (scene A) | Frame time (scene B) | 1 min |
| Prompt engineering | Prompt + examples | Accuracy (eval set) | Accuracy (holdout set) | 30 sec |

## What the Agent Needs

- **Autonomy**: Runs until session limit without asking permission.
- **Judgment**: Decides whether to fix a crash or abandon the idea. Uses provisional commits when a two-step improvement seems promising. Archives the arc and starts fresh when stuck — reads the journal and archived branch before re-approaching.
- **Memory**: Reads the experiment log and journal before proposing ideas. Doesn't retry failures without justification.
- **Reasoning**: Runs the micro-manifold before each experiment. Writes hypotheses, consults the record, challenges assumptions. This is not a formality — it's how the agent externalizes its thinking and how the human evaluates the quality of the search.
- **Restraint**: Not every metric improvement is worth keeping. Simplicity is a judgment call. The human reviews diffs to verify.
- **Honesty**: Records surprises in the journal. Notes when a hypothesis was wrong. Red-teams its own successes. Flags concerns rather than rationalizing them away. Doesn't hide doubt behind a clean metric.

## What the Human Does

- **Writes `program.md`** — the strategic layer.
- **Reviews results** — `results.tsv` for metrics and outcomes. `journal/` for reasoning quality. Both matter.
- **Steers mid-session** — `strategy_override.md` for real-time redirection. Check `override_ack` rows to verify interpretation.
- **Composes branches** — evaluates multi-agent results, decides what to combine.
- **Manages arcs** — reviews archived research arcs. Decides whether to delete, cherry-pick, merge, or keep for reference. The nuke is a human decision, not an agent decision.
- **Reviews code** — the protocol catches numerical failures, but human review of the experiment surface remains the ultimate sanity check. LOC tracking in session summaries provides a signal for when review is overdue.
- **Reviews the journal** — the journal reveals whether the agent is reasoning or mutating randomly. Consistently wrong hypotheses, absent NODES/REFLECT entries, or superficial reflections are signals that the agent's search quality is degrading.
- **(Optional) Configures hardening** — sets thresholds for acceptance, holdout divergence, parameter bounds, and thrash limit when domain expertise permits.

## Properties

- **Anytime**: Current best is always a valid, committed result.
- **Monotonic**: Branch advances only on improvement. Single provisional commits are the bounded exception, resolved within one experiment.
- **Auditable**: Every experiment is a git commit with a structured log entry — metrics, hypothesis, status, description. The journal provides the reasoning behind each entry.
- **Hardware-adaptive**: Fixed budget optimizes for your machine.
- **Steerable**: Human redirects mid-session via strategy override, with acknowledgment.
- **Self-correcting**: Arc archiving on thrash. Session rotation prevents context rot. Failure memory prevents redundant exploration. The journal ensures fresh arcs inherit understanding.
- **Legible**: The micro-manifold, red-team, and journal make the search process reviewable. The human can evaluate not just *what* the agent did but *how it was thinking* — and whether it questioned its own successes.
- **Zero-config**: Base protocol works out of the box. Hardening tier available when the user is ready.
- **Non-destructive**: Nothing is destroyed during the search. Archived arcs preserve code; the journal preserves reasoning. The human decides what to prune after review.

## Limitations

### Search, not research
The micro-manifold and post-experiment red-team move the protocol from random search toward deliberate, self-skeptical search. The agent reasons before acting and questions its own successes before building on them. But it still isn't designing controlled experiments or building causal models. The gap between search and research is narrowed significantly — not closed.

### Fixed evaluation surfaces are gameable
Holdout metrics with automated response reduce overfitting significantly but don't eliminate it. Over enough iterations, the agent can find solutions that perform well on both sets while still specializing. The human reviewing actual model behavior — not just numbers — is the last line of defense.

### Simplicity is still a judgment call
LOC tracking gives visibility. Human diff review gives enforcement. But between reviews, the agent can add complexity freely. Over 100 experiments, review fatigue is real.

### Journal quality depends on agent quality
The journal is only as good as the agent's reasoning. A weak agent produces superficial NODES and REFLECT entries that provide no value. The human should sample journal entries periodically to verify reasoning quality. If the journal reads like filler, the micro-manifold isn't working and the agent may need a better program.md or a more capable model.

### Arc archiving accumulates branches
Each thrash event creates a new archived branch. Over a long research campaign, this can produce many arcs. The human must periodically review and prune — deleting dead-end arcs, cherry-picking valuable commits, or merging insights. Without curation, the branch namespace becomes cluttered. This is a maintenance cost, traded against the benefit of preserving everything.

### Threshold calibration requires domain expertise
The hardening tier's thresholds are genuinely hard to set. The calibration guidance helps, but there's no substitute for understanding your metric's variance. This is why the hardening tier is optional — it's better to run without thresholds than to run with wrong ones.

## Design Rationale

This protocol went through six iterations:

1. **Original (Karpathy)**: 3 files, 8-step loop, zero configuration. Elegant, effective, structurally blind to 12 failure modes.

2. **First hardening**: 5 components, 6 auxiliary files, 13-step loop, 10+ parameters. Addressed all 12 failure modes but introduced 13 new ones — over-parameterized, fragile file dependencies, agent judgment embedded in every guardrail, more complex than the thing it was guarding.

3. **Triage**: Back to 3 components, 2 auxiliary files, 9-step loop, 3 parameters. Kept the 5 highest-impact mitigations, cut 7. But presented the remaining 3 parameters as simple configuration when they're actually hard design decisions, cut too aggressively in two places (provisional commits, session handoff), and dismissed the search/research gap without exploring lightweight mitigations.

4. **LMM refinement**: Applied the Lincoln Manifold Method to the triaged protocol. Core insight: over-invested in runtime constraints (which require calibration) and under-invested in observability tools (which work out of the box). Split into base protocol (zero config) and optional hardening tier. Restored single provisional commit and session summary rows. Added hypothesis tracking.

5. **Micro-manifold integration**: Embedded a compressed LMM cycle into the experiment loop itself. Added the journal as a persistent reasoning artifact. The journal solved the baseline reset problem — resets destroy code, not knowledge — and made the search process legible to the human.

6. **Arc archiving + red-teaming keeps (this version)**: Replaced baseline reset with branch archiving. When the agent gets stuck, it preserves the current arc as a named branch and starts fresh from baseline. Nothing is destroyed — the code survives in the archive, the reasoning survives in the journal, and the human decides what to prune after review. The nuke is still available; it's just a human decision at review time, not an agent decision under pressure.

Added post-experiment red-teaming on every `keep` — the agent scrutinizes its own successes before building on them, flagging concerns that travel with the improvement across sessions. This closes the asymmetry where the protocol was skeptical of ideas (pre-experiment manifold) but credulous of results (if number go down, keep).

The principle holds: if the guardrails need guardrails, cut the guardrails. The micro-manifold and red-team aren't guardrails — they're *cognition*. Arc archiving isn't a guardrail — it's *preservation*. They don't constrain the agent's behavior; they improve the agent's thinking and protect the human's ability to make informed decisions about what to keep.

## Origin

Original design by Andrej Karpathy (March 2026). Protocol analysis, red-teaming, and hardened design by A.N. Josserand-Austin and Claude (March 2026). Reasoning layer adapted from the Lincoln Manifold Method by A.N. Justin.
