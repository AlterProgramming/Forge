# Forge agent instructions

## Recover Qwen execution before creating new infrastructure

**Canonical Qwen execution memory:** [docs/QWEN_EXECUTION_SETUP.md](docs/QWEN_EXECUTION_SETUP.md).

For any Qwen3-0.6B, agent-communication, LoRA training, or compositional inference work in this repository:

1. Read `docs/QWEN_EXECUTION_SETUP.md` and inspect the current Actions run and adapter artifact **before** proposing or executing a new setup.
2. Reuse the actual base `Qwen/Qwen3-0.6B` and trained PEFT LoRA checkpoint; do not treat a prompt wrapper as trained weights.
3. Evaluate outputs by completed downstream tasks; do not stop because of large token or compute cost alone. Respect actual runtime constraints.
4. Connect existing learned primitives through real outputs (never gold substitutions); preserve full traces and negative results.
5. Record code branch, run ID, adapter SHA, verified metrics, and supersession after any new experiment.

The newest verified source/weights and exact training/evaluation commands are in the linked runbook. Code is on research branches rather than on `main`; checkout the branch named in the runbook to execute it.
