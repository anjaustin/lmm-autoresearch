# autoresearch

This is an experiment to have the LLM do its own research — deliberately, not randomly.

For the full protocol rationale, design history, and hardening analysis, see `AUTORESEARCH.md`.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autoresearch/<tag>-arc1` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>-arc1` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `AUTORESEARCH.md` — the protocol: philosophy, loop, constraints, hardening.
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data prep, tokenizer, dataloader, evaluation. Do not modify.
   - `train.py` — the file you modify. Model architecture, optimizer, training loop.
4. **Verify data exists**: Check that `~/.cache/autoresearch/` contains data shards and a tokenizer. If not, tell the human to run `uv run prepare.py`.
5. **Initialize results.tsv**: Create `results.tsv` with the header row:
   ```
   commit	primary	holdout	status	hypothesis	description
   ```
6. **Create strategy_override.md**: Create an empty `strategy_override.md` file.
7. **Create journal directory**: `mkdir -p journal/`
8. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU. The training script runs for a **fixed time budget of 5 minutes** (wall clock training time, excluding startup/compilation). You launch it simply as: `uv run train.py`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, data loading, tokenizer, and training constants (time budget, sequence length, etc).
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the evaluation harness. The `evaluate_bpb` function in `prepare.py` is the ground truth metric.
- Edit prior session journal files. The journal is additive only.

**The goal is simple: get the lowest val_bpb.** Since the time budget is fixed, you don't need to worry about training time — it's always 5 minutes. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size. The only constraint is that the code runs without crashing and finishes within the time budget.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful val_bpb gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.001 val_bpb improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 val_bpb improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Output format

Once the script finishes it prints a summary like this:

```
---
val_bpb:          0.997900
training_seconds: 300.1
total_seconds:    325.9
peak_vram_mb:     45060.2
mfu_percent:      39.80
total_tokens_M:   499.6
num_steps:        953
num_params_M:     50.3
depth:            8
```

Note that the script is configured to always stop after 5 minutes, so depending on the computing platform of this computer the numbers might look different. You can extract the key metric from the log file:

```
grep "^val_bpb:" run.log
```

## Sanity check

After each experiment, check for NaN/Inf in the output. If `val_bpb` is NaN, Inf, or missing from the output, the experiment has a numerical instability problem. Revert immediately regardless of any other signal.

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 6 columns:

```
commit	primary	holdout	status	hypothesis	description
```

1. git commit hash (short, 7 chars)
2. val_bpb achieved (e.g. 1.234567) — use 0.000000 for crashes
3. holdout metric (leave empty unless hardening tier is enabled)
4. status: `keep`, `discard`, `crash`, `provisional`, `recovered`, `session_summary`, `override_ack`, or `arc_archived`
5. hypothesis: one-line prediction written BEFORE the experiment (what you expect and why)
6. description: what actually happened

Example:

```
commit	primary	holdout	status	hypothesis	description
a1b2c3d	0.997900		keep	baseline run, establishing reference metric	baseline
b2c3d4e	0.993200		keep	higher LR should improve convergence given stable loss curve	LR 0.04 improved convergence as expected
c3d4e5f	1.005000		discard	GeLU may help late-training stability	GeLU worse — 4th failed activation swap, architecture likely coupled to ReLU-squared sparsity
d4e5f6g	0.000000		crash	doubling width should improve capacity	OOM — model too large for available VRAM
```

## The micro-manifold

Before each experiment, run a compressed reasoning cycle. This is structured thinking — not busywork.

**1. RAW — Form your hypothesis.** What do you expect to happen and why? Write it as one line — this becomes the `hypothesis` column in results.tsv.

**2. NODES — Consult the record.** Read results.tsv and the journal. What do prior experiments in this area of the search space show? What tensions exist between this idea and what's worked before? Write 2-5 lines to your session journal file.

**3. REFLECT — Challenge yourself.** Am I trying this because I have a reason, or because I ran out of ideas? What assumption am I making that might be wrong? Write 2-5 lines to your session journal file.

**4. SYNTHESIZE — Make the change.** The actual code modification, informed by the first three steps.

After the experiment, update the journal with what actually happened vs. what you expected. **Note surprises.** A wrong hypothesis is not a failure — it's a finding. Document what it tells you about the search space.

## Red-teaming successes

Every `keep` gets scrutinized before you move on. When an experiment improves the metric, do NOT blindly accept it. Immediately red-team the result in your journal (2-5 lines):

- **Is this genuine?** Could the improvement be stochastic noise? Is the magnitude above the variance you've observed in prior runs?
- **Is this robust?** Did you introduce a coupling, a shortcut, or a hidden dependency that will constrain future experiments?
- **Is this clean?** Did the code get simpler or more complex? Is the complexity justified?
- **What would break this?** If you change the architecture, optimizer, or batch size next, does this improvement survive?

If you identify a concern, mark it with `⚑ FLAG:` in the journal. Flags don't trigger reverts — the metric improved and the protocol respects that. But flags travel with the improvement. They're visible to you in future NODES phases and to the human during review.

**Watch for flag accumulation.** If three or more consecutive kept experiments carry flags, note this in your REFLECT phase. The branch may be building on a fragile foundation. Consider whether to continue, archive the arc, or alert the human via a journal note.

## Documenting failures

Failures are structural scaffolding, not waste. When an experiment fails, don't just log "discard" and move on. Ask: what does this failure *tell me*?

- A single failure is data.
- A pattern of failures in the same area is a *discovery* — evidence of a constraint, a coupling, or a wall in the search space.
- The shape of what doesn't work reveals the shape of what will.

Your journal should accumulate a map of the search space — not just where the peaks are, but where the walls are, and *why* they're walls.

## The journal

One file per session: `journal/session_NNN.md`

Each file contains:
- Micro-manifold entries for each experiment (NODES + REFLECT, 2-5 lines each)
- Result annotations (what happened vs. hypothesis, surprises)
- Red-team entries for kept experiments (2-5 lines, with ⚑ FLAGS for concerns)
- A session-end reflection

The journal is your reasoning made visible. It survives session boundaries and arc archives. When you start a new session or a new arc, the journal is how you inherit understanding from your past work.

**Journal hygiene:**
- Write during the session, not retroactively
- Never edit prior session files
- If the journal grows large, read the most recent 2-3 sessions in full and scan session-end reflections from older sessions

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar5-arc1`).

### Session start

1. Read `program.md`, `strategy_override.md`, and `results.tsv`
2. Check git state: current branch, current best commit
3. Read `session_summary` rows in results.tsv for compressed history
4. Read recent journal files (last 2-3 sessions)
5. Create `journal/session_NNN.md` for this session
6. If starting a new arc: read journal and optionally inspect the archived branch for reference

### Experiment loop

LOOP UNTIL SESSION LIMIT (default: 20 experiments):

1. Check `strategy_override.md` for new human instructions. If changed, log an `override_ack` row to results.tsv with your interpretation of the override.
2. **Micro-manifold**: RAW (hypothesis) → NODES (consult record) → REFLECT (challenge assumptions) → SYNTHESIZE (code change). Write NODES and REFLECT to journal.
3. git commit
4. Run the experiment: `uv run train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^val_bpb:\|^peak_vram_mb:" run.log`
6. **Sanity check**: If val_bpb is NaN, Inf, or missing → revert, log as crash, move on.
7. **Accept/reject**:
   - Primary improved → keep the commit, advance the branch
   - Primary not improved + no provisional outstanding → revert
   - Primary not improved + provisional available → mark as `provisional` (one max, no chaining)
   - Next experiment after provisional: recovered → keep both (`recovered`); not recovered → revert both
8. Log to results.tsv (commit, primary, holdout, status, hypothesis, description)
9. Update journal with result vs. hypothesis. Note surprises. If this was a failure, document what the failure tells you about the search space.
10. **If kept → Red-team the result.** Write 2-5 lines in the journal. Is this genuine? Robust? Clean? What would break it? Flag concerns with `⚑ FLAG:`. Do NOT blindly accept an improvement — scrutinize it before building on it.
11. **Thrash check**: If you're stuck (many consecutive reverts, no productive direction), archive the current arc and start a new one (see below).

### Archiving a research arc

When you're stuck — not just a few reverts, but genuinely out of productive ideas in the current direction:

1. Tag or rename the current branch (e.g. `autoresearch/mar5-arc1`)
2. Log an `arc_archived` row to results.tsv with the archived branch name, best metric, and reason
3. Create a new branch from the baseline commit (e.g. `autoresearch/mar5-arc2`)
4. Read the journal for accumulated understanding
5. Optionally inspect the archived branch's code via `git diff` or `git show`
6. Begin the new arc with informed hypotheses — you know where the walls are

**Do not delete archived branches.** The human decides what to prune after reviewing the arcs.

### Session end

1. Write a session-end reflection to your journal:
   - What worked and why
   - What didn't — and what the failures revealed about the search space
   - Emerging patterns
   - Current best commit and metric
2. Append a `session_summary` row to results.tsv:
   - Current best commit hash and primary metric
   - Session stats: keeps / discards / crashes
   - Current LOC of `train.py` (run `wc -l train.py`)
3. The session ends. The next session reads results.tsv, the journal, and the branch.

### Crash handling

If a run crashes (OOM, bug, etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, log it as `crash`, document in the journal what the crash tells you, and move on.

### Timeout

Each experiment should take ~5 minutes total (+ a few seconds for startup and eval overhead). If a run exceeds 10 minutes, kill it and treat it as a crash (revert and log).

## Strategy override

The human can steer you mid-session by editing `strategy_override.md`. Check this file at the start of each loop iteration. If it's been modified:

1. Read the new instructions
2. Log an `override_ack` row to results.tsv with a brief statement of how you're interpreting the override
3. Incorporate the instructions into your approach

Examples of overrides:
- "Stop trying activation functions. Focus on optimizer changes."
- "Holdout is diverging — prioritize generalization."
- "Archive this arc. Try a completely different architecture family."

## Session limits and autonomy

**Session limit**: Run approximately 20 experiments per session (~2 hours). At the end, write your session-end reflection and summary, then stop.

**NEVER STOP MID-SESSION**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep or away from the computer. You are autonomous until session end. If you run out of ideas, read the journal. Re-read the in-scope files. Look at patterns in your failures. Try combining ideas that individually showed promise. If you're genuinely stuck across multiple approaches, archive the arc and start fresh — don't stop.

As an example use case, a user might leave you running while they sleep. With sessions of ~20 experiments each, a human can chain sessions to run approximately 100 experiments overnight. The user wakes up to results.tsv, the journal, and a set of research arcs — all completed by you while they slept.

## Provisional commits

You can keep ONE experiment that didn't improve the metric, marked `provisional`, if you believe the next experiment will recover. Rules:

- Maximum one provisional commit at a time. No chaining.
- If the next experiment recovers (metric improves past the pre-provisional best), both are kept. Log the recovery as `recovered`.
- If the next experiment does not recover, revert BOTH the provisional and the follow-up.
- Use this sparingly — only when you have a specific, articulable reason to believe a two-step improvement is in play.
