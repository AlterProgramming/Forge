# State-dependent rewrite causal diagnostic — 2026-11-16

## Executed locally (CPU)
Command: `python rewrite_state_dependence.py --steps 350 --seeds 2`; PyTorch 2.10.0+cpu, 2 CPU threads; elapsed 14.915 seconds. Source code committed on this branch. Local output file contained all per-length/ablation values; the table below preserves the primary L24 results. **The committed source differs from the local script only in the destination of the output JSON** (relative working directory instead of /mnt/data).

Training: 256 synthetic sequences of length 6 per task/seed, AdamW lr 0.003, gradient clip 1, BCE at all timesteps, 350 full-batch steps, fixed generator seeds 100/101 for training; test generator seeds 1024/1025, 512 sequences at each length 6,12,24. Network: 9,156 parameters, encoder 10→64→32, delta 64→64→32, linear 32→4 readout. Same parameter count, optimizer, examples, training steps, and forward-pass network cost in both modes. State-dependent mode computes s'=s+f(s,e); state-blind mode computes s'=s+f(0,e). Readout is applied after each update. Interventions: force f(0,e) at inference, swap state across examples halfway through the sequence, or disable updates after step 3. No external solver.

| Task | Mode | Seed | L24 accuracy | Blind intervention | Midpoint state shuffle | Stop after 3 |
|---|---|---:|---:|---:|---:|---:|
| overwrite | dependent | 0 | .85280 | .56783 | .80802 | .62738 |
| overwrite | dependent | 1 | .87577 | .57489 | .82729 | .62852 |
| overwrite | blind | 0 | .74571 | .74571 | .71637 | .61928 |
| overwrite | blind | 1 | .74449 | .74449 | .72392 | .62299 |
| toggle | dependent | 0 | .76119 | .56154 | .72078 | .61928 |
| toggle | dependent | 1 | .86066 | .56230 | .74847 | .62321 |
| toggle | blind | 0 | .58197 | .58197 | .56659 | .61485 |
| toggle | blind | 1 | .57817 | .57817 | .56533 | .61471 |

L6 unablated: overwrite dependent .99967/.99927, blind .94491/.94645; toggle dependent .99780/.99959, blind .86458/.85734. L12 unablated: overwrite dependent .97941/.98177, blind .83525/.84513; toggle dependent .93518/.98694, blind .63493/.62956.

## Interpretation and limitations
This supplies direct causal evidence that the *small synthetic controller* uses its previous state to compute the next update, not merely to retain an accumulated sum. For the 24-step toggle task, mean accuracy dependent .81093 versus blind .58007; overwrite .86429 versus .74510. The state-blind model has the same total trainable parameters but fewer input-dependent arithmetic operations because its state-input branch is zero. These are only two seeds, synthetic tasks with fixed four-key semantics, and a length shift within the same generators. No Qwen weights were loaded, no Qwen hidden states were read, no frozen Qwen readback was tested, and no robust cross-domain improvement is established. The prior recurrent-replacement model outperformed additive rewriting in the earlier diagnostic; this result does not supersede that comparison.

Next obstruction: make a frozen Qwen3-0.6B decoder consume the updated learned state at every step, compare to equal-cost recurrent replacement and state-blind additive controls, then measure causal state interventions on independent held-out tasks. The existing SHA-verified checkpoint artifact workflow is required in a network-enabled runner.
