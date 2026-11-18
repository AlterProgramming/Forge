# Learned readback causal diagnostic — 2026-11-18

Executed locally on CPU, not Qwen3-0.6B. Parent branch: research/qwen3-mixed-rewrite-replacement-20261117. Earlier results left untouched.

## Reproduction
Run `python experiments/qwen3-context-rewrite/rewrite_readback_20261118.py --steps 300 --seeds 3 --threads 2` from a writable directory. Python with PyTorch 2.10.0+cpu; Linux 6.18.57 x86_64. Script writes full JSON (all tasks, lengths and ablations). CSV in this branch records seed-level metrics for mixed task and 24-step transfer; CSV floats rounded to six decimals. RNG seeds and exact split generation are specified in source. Fixed training: 256 independently seeded 6-step mixed sequences per seed, reused for 300 AdamW updates (lr 0.003, gradient clip 1). Test: 512 newly seeded sequences per seed, length 6/24/48, mixed/overwrite/toggle. No train/test sequence reuse by construction. 3 seeds, modes rewrite/additive, blind/additive, replacement. Each has 10,212 parameters and the same architecture; only state update/readback behavior changes. All readbacks are computed even when zeroed, making nominal forward graph comparable but not measured FLOPs. Runtime 16.059 seconds for all nine trainings/evaluations on two CPU threads; no energy, FLOPs, peak RSS, or GPU costs measured.

## Results
Mean held-out bit accuracy across 3 seeds:

| mode | mixed 24 | mixed 48 | overwrite 24 | toggle 24 |
|---|---:|---:|---:|---:|
| additive REWRITE + learned readback | 0.8465 | 0.7294 | 0.9027 | 0.7554 |
| additive state-blind | 0.6727 | 0.6152 | 0.7403 | 0.5914 |
| recurrent replacement + learned readback | **0.9873** | **0.9774** | **0.9959** | **0.9538** |

Causal interventions on additive REWRITE, mixed L24: normal 0.8465; zero readback 0.5809; shuffled readback across examples 0.6736; one-step lag 0.7378. At L48: normal 0.7294; zero 0.5407; shuffle 0.5959; lag 0.6741. Thus useful dependence on repeatedly updated state is demonstrated *in this synthetic network*. Recurrent replacement is decisively stronger.

## Interpretation / obstruction
This does not test Qwen's ability to read rewritten latent state. The 'readback' here is an auxiliary trainable MLP, not a Qwen attention read or Qwen-conditioned operation. It is not a general representation rewrite: tasks use four fixed keys, deterministic binary updates and supervised intermediate state labels. Matched parameter counts do not imply matched FLOPs, and readback is computed redundantly in the blind mode. Results do not establish independent semantic held-out generalization or an advantage over recurrent replacement. Next obstruction is inserting learned latent state into frozen Qwen at every step, then evaluating output-only training and matched frozen-Qwen baselines across unrelated tasks. No claim of Qwen capability is warranted.
