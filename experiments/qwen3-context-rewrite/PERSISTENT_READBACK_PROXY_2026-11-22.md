# Persistent readback proxy diagnostic — 2026-11-22
Executed locally using torch 2.10.0+cpu; 4.7233 s wall time, 2 CPU threads. Run: `python experiment.py --steps 300 --output results.json`.
Each model: 2292 trainable parameters, 300 AdamW steps at lr .002, minibatch 64, training length 6, 512 train examples per seed, three seeds (13,29,47). Independent 512-example test splits at lengths 6,24,48, with exact seed rules in code. Two keys; event set-0/set-1/toggle/no-op; final two-bit state as four-way target. Final-answer supervision only.
The latent workspace s persists separately from readback r=tanh(1.7s). Both arms call the same update and fixed readback per event. REWRITE uses s+delta, replacement uses delta and zeros its direct s input, but retains the readback path; this is a **stateful replacement control**, not state-blind.
Mean exact final-state accuracy (3 seeds):
| arm | L6 | L24 | L48 |
|---|---:|---:|---:|
| additive rewrite | 0.6712 | 0.3802 | 0.2910 |
| stateful replacement | 0.9023 | 0.8991 | 0.9063 |
At L24, shuffling readback among examples: rewrite 0.3672, replacement 0.3812; zeroing readback: rewrite 0.3454, replacement 0.3418. The stateful replacement suffers large causal losses; additive rewrite is weak and does not reliably benefit from readback. Seed-level results are in results.json.
**Negative finding:** explicitly separating persistent state from fixed readback does not rescue additive rewrite on long sequences. Replacement strongly dominates. No frozen Qwen checkpoint was loaded: the fixed tanh readback is only a diagnostic proxy, not evidence of Qwen rewriting its representation. Not a general architecture test. Parameter and number-of-update calls matched; full FLOPs, memory and energy costs were not measured; 4.72s CPU wall time measured. Prior branches remain unchanged.
**Next obstruction:** actual frozen Qwen readback with an independent persistent workspace, then equal-compute stateful replacement and state-blind controls; evaluate independent tasks and length OOD.