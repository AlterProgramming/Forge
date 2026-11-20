# Frozen-Qwen readback run: completed result and independent audit (2026-11-20)

Source: https://github.com/AlterProgramming/Forge/actions/runs/44637769586
Artifact: https://github.com/AlterProgramming/Forge/actions/runs/44637769586/artifacts/14585197060
Source commit: 757af9b71b3bc760f49823572fb7296ad409df82
Job 135960638031 completed successfully on 2026-11-19 at 23:32 UTC.

The experiment executed frozen official Qwen3-0.6B with a trainable 199,746-parameter controller, outcome-only labels, 12 training steps, 48 training examples, and 24 test examples per task/length. CPU workflow used two threads. The training run took approximately five minutes including dependency installation and checkpoint acquisition; this is not a precise isolated compute measurement.

| Task | Stateful len3 | Stateful len6 | Blind len3 | Blind len6 | Stateful ablated len6 |
|---|---:|---:|---:|---:|---:|
| parity | 13/24 | 13/24 | 13/24 | 13/24 | 11/24 |
| last_write | 12/24 | 11/24 | 9/24 | 13/24 | 13/24 |

Executed independent audit: `python experiments/qwen3-context-rewrite/audit_readback_20261120.py`.
For len6, both tasks' test labels have 13 zeros and 11 ones; predicting zero always gets 13/24. Parity len3 has the same 13/24 majority. Last-write len3 is balanced (12/12). All len6 results are at or below the constant-majority predictor. The 24 length-six test sequences contain only 18 distinct sequences; length-three tests contain all 8 possible sequences, with 16 duplicates. Exact two-sided binomial p vs chance for 13/24 or 11/24 is 0.83882 (not an appropriate paired model comparison). No evidence of learned Qwen readback advantage.

**Important causal flaw:** the `ablate` flag zeroes the state after every Qwen pass, so this intervention changes the entire downstream input distribution and is not a matched-compute causal isolation of readback. Also, `state=state+self.update(...)` followed by `state=Qwen(state)` does not preserve a separately writable state across steps. The blind control still receives Qwen's previous readback through the residual `state` term, so it is *not* a true state-independent control. The readout is an external trainable linear head, not Qwen token generation. This prevents interpreting the experiment as evidence of a trainable latent REWRITE operation that Qwen itself uses.

Next obstruction: define an actually independent state-blind update, preserve a distinct writable workspace and Qwen readback, match FLOPs, and train across enough independent seeds and task instances for paired comparisons. Do not claim success from this run. Prior research branches and results unchanged.
