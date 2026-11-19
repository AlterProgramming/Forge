# Frozen Qwen latent readback feasibility — 2026-11-19

## Status
GitHub Actions run https://github.com/AlterProgramming/Forge/actions/runs/44637769586 started. **No outcome results were available at initial publication.** Do not interpret an in-progress workflow as a completed experiment.

## Relationship to earlier research
New branch from research/qwen3-readback-causal-20261118. Prior experiments remain unchanged. The source and workflow are separate additions. Earlier arithmetic and bAbI outcomes are not superseded.

## Intervention
Frozen official Qwen/Qwen3-0.6B consumes a trainable continuous latent state via inputs_embeds after each update. The next update uses Qwen's last-layer readback. The trainable controller is an MLP mapping concatenated prior state and event embedding to a residual update; a two-class readout is trained from final answers only. No intermediate state labels.

## Controls and limitations
Matched parameter-count, identical compute-path control: previous state input to the update MLP is zeroed (state-blind), but the recurrent Qwen readback remains. Thus this is **not** a complete recurrent-replacement control. Ablation zeroes the Qwen readback at each step, but does not match its FLOPs; do not infer causal benefit from a trivial loss of information alone. Independent tasks: parity and last-write over A/B events; train sequences length 3, evaluate lengths 3 and 6. These are simple synthetic tasks, not a general capability test.

Seed 119 for weights; train generator seed 100, 48 training examples/task; held-out generator seeds 10003 and 10006, 24 examples per length/task. AdamW lr 0.0002, 12 steps per arm/task, cross-entropy on final outcome only, batch size 1, float32 frozen Qwen, CPU 2 threads. Note the 12-step budget only uses the first 12 of 48 generated training examples, and **cannot support a robust positive claim**. Four training runs (2 tasks x 2 arms). All code is in qwen_latent_readback_20261119.py. The code reports trainable parameters and wall-clock duration; peak memory, FLOPs, energy, and full model download cost are not yet instrumented. No pre-registered success threshold.

## Interpretation
This run tests whether the frozen Qwen hidden-state readback can be incorporated into a differentiable outcome-trained update loop at all. A successful workflow without accuracy gains is only an engineering milestone. Next obstruction: optimize runtime, complete matched replacement/readback-shuffle controls, repeat seeds and independent tasks, and measure full costs before any generalization claim.
