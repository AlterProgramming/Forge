# Qwen3 general-purpose two-pass inference

The reusable CLI is `qwen_boost_cli.py`. It runs the official **frozen**
`Qwen/Qwen3-0.6B` model, or your own downloaded local checkpoint, on
**arbitrary user prompts**.

It does **not** retrain Qwen and does **not** contain a bAbI graph solver.

## What it does

1. Generates one Qwen3 native-thinking answer, up to a token budget (512 by default).
2. If the thinking pass reaches a complete response, returns that answer.
3. If Qwen uses its thinking budget without finishing, reuses its partial
   computation as a scratchpad and calls the **same model** in non-thinking
   mode to finalize the answer (up to 192 more tokens).
4. Optionally generates an ordinary non-thinking baseline for comparison,
   and records all tokens, output traces, and timing.

The intervention is demonstrated on the **exploratory** GSM8K/ARC/MBPP
samples in PR #4: 3/12 correct with native thinking alone versus 7/12
with forced completion. The finalization required just 199 additional
generated tokens on that sample. These examples were repeatedly
examined while developing the procedure: **this is not yet evidence of
improvement on arbitrary unseen prompts**. Independent held-out validation
is recorded in PR #5.

## Run on a Mac with Qwen3 downloaded locally

```bash
python -m pip install 'torch>=2.4' 'transformers>=4.51,<5' accelerate safetensors
python qwen_boost_cli.py 'Explain the purpose of a software dependency graph.' \
  --model /path/to/your/Qwen3-0.6B \
  --device mps \
  --compare \
  --json-output trace.json
```

For CPU, use `--device cpu`; for auto device detection, omit
`--device`. The model directory must be a valid Transformers checkpoint
with a Qwen3 chat template, tokenizer, and the original safetensors.

The `--answer-format` option can be `text` (default), `number`,
`choice`, or `python`. If omitted, the tool attempts a normal answer
rather than forcing a domain-specific response structure.

```bash
python qwen_boost_cli.py 'Write a function to sort strings by length.' \
  --model Qwen/Qwen3-0.6B --answer-format python
python qwen_boost_cli.py --self-test
```

The tool does not execute model-generated code, and a successful final
answer is not automatically a correct one. Inference-time improvement
requires additional compute; the precise cost is logged in the JSON output.
