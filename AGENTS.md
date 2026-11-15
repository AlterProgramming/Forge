# Forge agent instructions

## Session zero: recover Qwen through Agent Command Center

Before declaring that Qwen weights, inference, adapters, or a runnable agent are unavailable, initialize through `AlterProgramming/Agent-Command-Center` (registered Forge adapter):

```bash
python3 "$COMMAND_CENTER_ROOT/scripts/session_bootstrap.py" \
  --session-id SESSION_ID --repository AlterProgramming/Forge \
  --scope qwen-execution --text "Recover existing Qwen3-0.6B execution" \
  --output SESSION_RECEIPT.json
```

When the Command Center checkout is absent, use the connected GitHub tools to read its `main` branch registries: `memory/registry.json`, `capabilities/registry.json`, and `sessions/registry.json`. Resolve **`capability.qwen-execution-recovery.v1`** and follow the canonical runbook and adapter artifact references in that capability before deciding inference is unavailable. Retain the bootstrap receipt in the handoff.

A missing local model cache is NOT proof that Qwen cannot run: the verified execution path is Forge GitHub Actions. The recorded LoRA `adapter_model.safetensors` comes from Actions run `43657885404` (artifact `14140515504`), and the base model is `Qwen/Qwen3-0.6B`. Verify artifact availability and never confuse this LoRA with unrelated research `.pt` files. Command Center recovery CI: https://github.com/AlterProgramming/Agent-Command-Center/actions/runs/43668126943 .

## Recover Qwen execution before creating new infrastructure

**Canonical Qwen execution memory:** [docs/QWEN_EXECUTION_SETUP.md](docs/QWEN_EXECUTION_SETUP.md).

For any Qwen3-0.6B, agent-communication, LoRA training, or compositional inference work in this repository:

1. Read `docs/QWEN_EXECUTION_SETUP.md` and inspect the current Actions run and adapter artifact **before** proposing or executing a new setup.
2. Reuse the actual base `Qwen/Qwen3-0.6B` and trained PEFT LoRA checkpoint; do not treat a prompt wrapper as trained weights.
3. Evaluate outputs by completed downstream tasks; do not stop because of large token or compute cost alone. Respect actual runtime constraints.
4. Connect existing learned primitives through real outputs (never gold substitutions); preserve full traces and negative results.
5. Record code branch, run ID, adapter SHA, verified metrics, and supersession after any new experiment.

The newest verified source/weights and exact training/evaluation commands are in the linked runbook. Code is on research branches rather than on `main`; checkout the branch named in the runbook to execute it.
