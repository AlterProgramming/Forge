# Qwen3 Learned Computational Adaptation — code and checkpoints

**Status: experimental research artifact (November 13, 2026).** This is the actual learned-parameter set and source produced in the current Qwen3-0.6B research session. It is **not game-specific**; bAbI spatial/path problems were the testbed. It does **not** claim universal transfer to projects or turn-based games.

## What is supplied

- `checkpoints/joint_graph_halt_curriculum_seed61426.pt` — **recommended best-performing research checkpoint**, learned graph construction and adaptive STOP policy, **2,958/3,000** on the held-out one–three-hop test questions and **616/628** on unseen four-hop questions under its own seeded evaluation. A separate paired rerun gave 2,955/3,000 and 618/628.
- `checkpoints/joint_graph_halt_curriculum_seed61425.pt` and `seed61427.pt` — two additional independent training seeds (both much weaker). Do not hide initialization instability.
- `checkpoints/joint_graph_halt_permuted_seed61425.pt` and `seed61426.pt` — Qwen-feature permutation controls.
- `checkpoints/joint_graph_halt_seed61425.pt` — no-curriculum control.
- `checkpoints/lora_qwen_stop.pt` and `frozen_qwen_stop.pt` — a *separate* experiment that trained rank-4 LoRA inside Qwen3’s last transformer layer, plus frozen-layer readout control. LoRA gave no improvement over the frozen control; these are not interchangeable with the graph checkpoints.
- `checkpoints/router.pt`, `blind_router.pt` — earlier selection-control checkpoints.
- `load_adaptation.py`, `vocab_150.json` — **inference-only loader** for joint graph checkpoints. Uses the exact model class from the executed training script, including a 150×1,024 frozen Qwen-feature buffer embedded in the checkpoint. No Qwen3 model download required for simple checkpoint inference over the known 150 sentence/question forms.
- `source/` — **verbatim original executed scripts**, including their absolute `/mnt/data/qwen_next` path assumptions; preserve these for provenance.
- `results/` and `REPORT_2026-11-13.md` — exact reported results, causal ablation, controls, and limits.

## Use it

From this directory, with PyTorch installed:

```bash
python load_adaptation.py --weights checkpoints/joint_graph_halt_curriculum_seed61426.pt
python load_adaptation.py --weights checkpoints/joint_graph_halt_curriculum_seed61426.pt --input example_input.json
```

`example_input.json` must contain `events` (a list of original bAbI-19 relation sentences) and `question` (one of the corresponding bAbI question forms). These are indexed against `vocab_150.json`; unsupported text is **not** accepted. Answers are direction paths of length 1–4. The model is **not** a text-generation/chat interface and not a directly attachable `PeftModel` LoRA.

## Original architecture and boundaries

Qwen/Qwen3-0.6B was the **frozen pretrained feature generator** during training. The learned joint adapter comprises eight anonymous relational-node addresses, 128/96-dimensional learned mappings, graph binding, a recursively applied traversal, and learned halt gates. The graph/tensor form, compass inverse operation, and basic traversal computation were **supplied**; the bindings and stop policy were trained from final answers alone. The best checkpoint trains ~301K parameters external to Qwen. It stores 150 frozen feature vectors derived from Qwen as a buffer. It does not repackage or replace Qwen's official 1.5GB checkpoint.

An exact classical graph solver solves all examples, and training was initialization-sensitive. Independent generalized capabilities remain unproven. These results are from an exploratory benchmark examined repeatedly, not a blind test.

## Portability

The inference loader is independent of the source scripts’ original absolute path. For exact retraining, use original benchmark data plus the saved Qwen sentence-feature extraction procedure, and update the original script's `P` path as needed. The archival research ZIP in this chat contains all experiment scripts, source data, results, and checkpoints; the published repository release prioritizes usable code plus original learned weight tensors.

## Files and provenance

A separate `MANIFEST_SHA256.json` gives exact hashes for binary checkpoints. **Neither a game-specific adapter nor a general-purpose self-creating model has been claimed.**
