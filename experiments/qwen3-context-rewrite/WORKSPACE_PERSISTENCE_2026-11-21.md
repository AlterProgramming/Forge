# Workspace persistence diagnostic — 2026-11-21

Executed locally using PyTorch 2.10.0+cpu, 2 CPU threads; 2.715346574783325 s wall time. Reproduce: `python experiments/qwen3-context-rewrite/workspace_persistence_probe_20261121.py --steps 220 --output results.json`.

**This is not Qwen3-0.6B execution.** It is an isolated, small-network failure-mode test prompted by the November 20 frozen-Qwen audit. Original research branches and results were not changed.

Architecture: 16-dimensional workspace; 8 event embeddings; MLP 32->32->16 update, separate MLP 16->16->16 readback and 4-way final-answer head. 2,324 parameters per arm. Rewrite receives prior state and adds a learned delta; blind receives zero prior state and replaces its workspace with an event-only update. Both execute the same layers, but actual FLOPs and memory were not profiled. Note that the separate readback does NOT feed the next update; this is not a demonstration of Qwen readback.

Data: keyed two-bit register with set-0, set-1, toggle, and no-op event types; final state classification only. 512 length-six training sequences for each seed (13,29,47), dataset seeds 1013/1029/1047; independent 512-example tests at lengths 6,24,48, generated with seeds 20006,20024,20048. AdamW lr .002, 220 minibatches of 64, cross entropy. No intermediate labels. Three independent initializations, same test examples for both arms.

| seed | arm | L6 | L24 | L24 shuffled | L24 zeroed | L48 |
|---|---|---:|---:|---:|---:|---:|
|13|rewrite|.615234|.402344|.312500|.373047|.246094|
|13|blind|.388672|.337891|.337891|.337891|.390625|
|29|rewrite|.533203|.337891|.291016|.386719|.300781|
|29|blind|.376953|.363281|.363281|.363281|.369141|
|47|rewrite|.615234|.281250|.287109|.378906|.255859|
|47|blind|.357422|.345703|.345703|.345703|.367188|

Mean rewrite L6 .58789, L24 .340495, L48 .267578; mean blind L6 .374349, L24 .348958, L48 .375651. At length 24 rewrite does not consistently beat blind; at length 48 it is worse in every seed. Interventions zero or shuffle the workspace after each step and can induce distribution shift; neither is a clean matched-compute causal proof.

**Interpretation:** learned additive state dependence can improve in-distribution training-length performance yet fail to preserve a useful state under repeated updates. This diagnostic does not establish general rewriting, Qwen latent integration, or robust OOD improvements. Main obstruction remains a persistent workspace that frozen Qwen actually reads after each rewrite, with true state-blind and recurrent-replacement controls, measured costs, and independent held-out tasks. No official checkpoint was loaded in this run.
