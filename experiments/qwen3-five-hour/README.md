# Forge Qwen3-0.6B five-hour continuation

- Starting weights: official `Qwen/Qwen3-0.6B` plus Forge's **existing** 393,216-parameter PEFT LoRA in `models/qwen3-0.6b-communication-lora-v2/`, verified by SHA-256.
- Study: continue training on the public **openai/gsm8k** grade-school mathematics training partition, with frozen disjoint development subset, and original Forge protocol rehearsal (1/5 updates).
- Select a best checkpoint by development teacher-forced token NLL. Never tune to the official test set.
- At completion, compare selected vs **original adapter** and frozen **base Qwen** on exactly the same official test prompts (numeric exactness and token NLL); compare old vs new on original Forge communication protocol, with full traces.
- Training is CPU-backed on GitHub Actions; the planned run has **17,400 seconds of training allocation** (setup and final evaluation separate). This is not proof of execution until a GitHub run actually starts.
- Persist real `adapter_model.safetensors`, its hash, exact split indices and hashes, full metrics and traces. Artifact is also committed to the research branch after success. Existing `main` weights are untouched.
- Improved means ONLY the `improvement_gate` in `results.json` passes: new heldout math NLL lower, exact correctness not lower, and protocol regression <= 1 of 64; this does **not** establish general reasoning or novel learned latent computation.
- A failed run or false gate is reported as such, not labeled as a better model.

Entry point: `python experiments/qwen3-five-hour/train.py --self-test`, or the branch workflow `qwen3-five-hour-continuation.yml`.
