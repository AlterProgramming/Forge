# Qwen Execution Setup — canonical project continuity record

**Recorded:** 2026-11-14 (America/Chicago).  
**Owner workspace:** AlterProgramming/Forge.  
**Status:** Executed training and verified inference, not a general-purpose Qwen capability breakthrough.  
**Critical rule:** First recover THIS setup and its latest artifacts before inventing new Qwen infrastructure or restarting training.

## Goal and principle

Use **real Qwen3-0.6B pretrained weights and real trained PEFT adapter weights** to execute agent communication experiments. Prefer operating on existing trained computational primitives; pass sender outputs to receiver inputs without inserting oracle-correct answers. Evaluate **completed downstream jobs** ahead of inference cost and token budgets. Never call an adapter "grokking" solely because its train loss decreases or it reproduces seen pairs. Save source hashes, exact prompt and response traces, train/test splits, and real safetensors.

## Model and runtime

- Base: Hugging Face `Qwen/Qwen3-0.6B`, using `AutoTokenizer` + `AutoModelForCausalLM` from `transformers`. Base weights stay frozen during PEFT/LoRA training.
- Inference: official Qwen chat template through `tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False, return_tensors="pt")`; greedy `generate(do_sample=False)` in the executed communication evaluations.
- Training: Python 3.11, `torch` CPU, `transformers>=4.51,<5`, `peft>=0.13,<1`, `accelerate`, `safetensors`. Verified runtime: Ubuntu GitHub-hosted Actions CPU runner; `torch.set_num_threads(2)` and float32 base-model load.
- Original learned adapter: PEFT LoRA **rank 8**, alpha **16**, dropout **0**, on `q_proj`, `v_proj`, `o_proj`, `down_proj` in **the last four decoder layers**. **393,216 trainable parameters**; `adapter_model.safetensors` + `adapter_config.json` are required to load it.
- A local Apple Silicon Mac copy of the pretrained base model was discussed; **its filesystem path is not established by these GitHub runs**. Do not invent local paths. GitHub Actions downloads the public base model at execution.

## Exact code and branches (these are NOT on main)

- First actual LoRA update: `research/qwen3-protocol-grokking-lora-20261114`, source `experiments/qwen3-context-rewrite/cross-domain/train_protocol_lora.py`, workflow `.github/workflows/qwen3-train-communication.yml`.
- **Continue the existing weights:** `research/qwen3-protocol-continuation-20261114`, source `experiments/qwen3-context-rewrite/cross-domain/continue_protocol_lora.py`, workflow `.github/workflows/qwen3-continue-protocol.yml`.
- **Execute learned primitive interactions:** `research/qwen3-protocol-composition-proof-20261115`, source `experiments/qwen3-context-rewrite/cross-domain/composed_handoff_demo.py`, workflow `.github/workflows/qwen3-composed-handoff.yml`.

## Actual executed runs and adapter artifacts

1. **First training:** https://github.com/AlterProgramming/Forge/actions/runs/43656857296 . Artifact **14139170088**, name `qwen3-0_6b-trained-communication-lora`. Train 384 steps at LR 0.0015. Heldout: SEND 1/16; RECEIVE 8/16; USE 0/16; complete 0/16. Real learned adapter, but sender and use collapsed.
2. **Continued training:** https://github.com/AlterProgramming/Forge/actions/runs/43657885404 . Artifact **14140515504**, name `qwen3-0_6b-continued-protocol-lora`. **THIS IS THE LATEST TRAINED CHECKPOINT**. Resumed first adapter with `PeftModel.from_pretrained(base, "previous/adapter", is_trainable=True)`; verified source adapter SHA256 prefix `02ec6abefbd0e27f`; trained 900 updates at LR 0.00045 with balanced SEND/RECEIVE/CALC and individual ENC_X/ENC_Y/DEC_X/DEC_Y rehearsal. Recorded adapter-weight delta L2 ~7.019. Heldout: SEND 15/16; RECEIVE 16/16; CALC direct 0/16; naive complete chain 0/16. The last result matters: no oracle substitutions, but direct learned Qwen CALC remains unlearned.
3. **Compose existing primitives:** https://github.com/AlterProgramming/Forge/actions/runs/43661450824 . Artifact **14140690763**, name `qwen3-actual-compositional-handoffs`. Actual Qwen ENC_X + ENC_Y sender and learned RECEIVE: **16/16 correct signals, 16/16 correct decoded values, and 16/16 correct deterministic sums** with no oracle correction. Direct joint SEND: 15/16 correct signals.
4. **Latest actual Qwen math follow-up:** https://github.com/AlterProgramming/Forge/actions/runs/43661880489 . This revises the original base-Qwen arithmetic receiver prompt to retry with symbolic expression and verify exact equations. On the same 16 already inspected withheld combinations: **atomic sender 16/16 correct signals, 16/16 decoded, 16/16 deterministic sum, 15/16 base Qwen math**; joint sender 15/16 signals and 14/16 base Qwen math. The base Qwen is invoked with the trained LoRA **disabled**. This is posthoc recovery, not a fresh blind sample or model training.

**Artifact retention:** GitHub Actions run artifacts are time-limited (workflow config uses retention-days 90). Keep a copy of the actual `adapter_model.safetensors` before expiration. A link or JSON report alone is **not** the adapter. The source repo does **not** contain the binary model weights.

## Reproduce / execute

GitHub-hosted:

1. Checkout branch `research/qwen3-protocol-composition-proof-20261115`.
2. Download artifact `qwen3-0_6b-continued-protocol-lora` from run `43657885404` to folder `previous/`; confirm files `previous/adapter/adapter_config.json` and `previous/adapter/adapter_model.safetensors`.
3. Install dependencies matching `.github/workflows/qwen3-composed-handoff.yml`.
4. From repo root run:

```bash
python experiments/qwen3-context-rewrite/cross-domain/composed_handoff_demo.py --self-test
python -u experiments/qwen3-context-rewrite/cross-domain/composed_handoff_demo.py \
  --model Qwen/Qwen3-0.6B --adapter previous/adapter \
  --out compositional_handoff_results.json
```

5. For continuation instead, checkout `research/qwen3-protocol-continuation-20261114` and run:

```bash
python -u experiments/qwen3-context-rewrite/cross-domain/continue_protocol_lora.py \
  --initial-adapter previous/adapter --model Qwen/Qwen3-0.6B \
  --steps 900 --eval-every 300 --lr 0.00045 --max-seconds 3200 \
  --out qwen_protocol_continued
```

The `continue_protocol_lora.py` expects *the first* adapter to resume as done in the verified continuation run; feeding the newest adapter produces a NEW follow-up and must be clearly labeled as such.

## Validation and limitations

- Fixed 8×8 two-attribute game; 48 combinations in training, 16 withheld via `(i+j)%4==0`; all constituent symbols occur in training. The code was **researcher-defined and factorized**, not discovered autonomously. These 16 test cases have been repeatedly inspected; later results are **posthoc** and not blind transfer evidence.
- Actual trained primitives: `ENC_X`, `ENC_Y`, `SEND`, `RECEIVE`, `CALC`. One learned shared adapter. `CALC` on new pairs remained 0/16 despite improvements on seen cases. The 16/16 full communication success used a deterministic arithmetic primitive; the most recent follow-up achieved 15/16 with **frozen pretrained Qwen arithmetic** after symbolic retry, *not* learned adapter CALC.
- Test **unseen tasks and agents** for communication transfer; avoid interpreting message sensitivity, training fit, or hand-coded channel as emergent language.
- Do **not** silently replace successful sender-produced outputs with gold intermediate values. Distinguish whether a result was verified by model, deterministic evaluator, or external oracle. Preserve prior complete artifacts unchanged. Use fresh run IDs and immutable hashes.
- User priority: **does the complete job get done?** Token counts and optimization come *after* correctness and compositional execution. No arbitrary early stopping due to token-cost concern. Practical resource limits and run timeouts still need honest reporting.

## Agent recovery procedure

1. Read this file first.
2. Inspect the latest completed Actions runs; recover code, logs and adapter artifacts. Verify whether a newer adapter has superseded the above.
3. Do an end-to-end smoke test with real pretrained Qwen + actual PEFT weights and exact message handoffs.
4. Locate first failed boundary: producing correct symbol, transmitting a typed result, decoding, selecting the next computation, or actually executing it.
5. Change and test the implicated existing primitive or its interaction *before* introducing another controller or wrapper.
6. Keep a clear record of **executed** vs **proposed** vs **unverified**, and commit the update to this runbook after proof.

This markdown file is a persistent **GitHub project continuity record**. It does not imply that ChatGPT account-level Memory has been directly updated.
