# Story-cluster robustness audit — 2026-11-14

## Executed
New analysis on **previously saved predictions**, not a new model training run. Uses the original story IDs to cluster the official held-out bAbI Task 19 questions. Two independently constructed evaluation sets (seeds 61425 and 61426), each with 3,000 1–3-hop questions from 1,000 stories and 628 four-hop questions. Script: `paired_cluster_audit.py`; full paired results: `paired_cluster_audit_results.json`.

- Evaluation seed 61425: curriculum 61425 versus no-curriculum 61425: 2,444 vs 1,039 / 3,000; difference +46.83 percentage points, story-cluster 95% bootstrap CI [44.73, 48.87] pp. On four-hop 325 vs 1 / 628.
- Evaluation seed 61425: curriculum 61425 versus permuted-Qwen-feature seed 61425: 2,444 vs 799 / 3,000; difference +54.83 pp, cluster CI [52.90, 56.70] pp. Four-hop 325 vs 0 / 628.
- Curriculum seed 61426 versus permuted features 61425: 2,955 vs 799 / 3,000; four-hop 618 vs 0 / 628.
- On independent evaluation seed 61426: curriculum 61425 vs no-curriculum 61425 is 2,472 vs 1,017 / 3,000; four-hop 326 vs 0 / 628. Curriculum 61426 vs permuted 61425 is 2,958 vs 806 / 3,000; four-hop 616 vs 0 / 628.
- All 12 story-level sign-flip tests: Monte Carlo p <= .001 with 1,000 sign-flip draws; numerical resolution is limited by that draw count.

## What this does NOT establish
This is **not** evidence for a general Qwen self-rewriting working representation. The existing joint model uses a task-specific differentiable graph, fixed four-step traversal and learned STOP head, with frozen Qwen sentence features; it does not rewrite Qwen's own latent working state. These contrasts do not guarantee matched total training compute, and permuting pretrained features can disrupt their learned geometry. Evaluation seeds resample from the same official held-out stories, so they are **not independent new test corpora**. There is no new Qwen forward pass or training in this audit. The strong paired differences are real for these saved predictions but should not be promoted to a general capability claim.

## Reproduce
From the accompanying 2026-11-13 research archive, extract `paired_test_predictions_eval_seed61425.json` and `paired_test_predictions_eval_seed61426.json` to `qwen3-learned-adaptation/results/qwen_joint_graph_halt_curriculum/` adjacent to the script's parent directory; run `python paired_cluster_audit.py` with Python 3 standard library. Exact source data, fixed seed 20261114, 1,000 resamples, paired question and story IDs, and all output rows are preserved. No original result files were modified.

## Next obstruction
A trained repeated state-dependent latent rewrite must feed its changed vectors back into the **same frozen Qwen** across two or more iterations, with outcome-only gradients and a matched non-rewrite control at equal Qwen forward passes and parameters. No such controlled demonstration was executed here.
