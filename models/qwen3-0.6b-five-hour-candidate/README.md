# Qwen3-0.6B 5-hour continuation

Base: `Qwen/Qwen3-0.6B`; source: Forge saved LoRA v2.
Selected adapter in `selected_adapter/`, which is the original if no validation checkpoint improved.
`results.json` includes disjoint splits, hashes, checkpoints, traces, costs, and a
strict improvement gate. Only a true gate warrants a narrow math-domain improvement claim.
Load with `PeftModel.from_pretrained(base, "selected_adapter")`.
