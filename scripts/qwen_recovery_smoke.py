#!/usr/bin/env python3
"""Cold-start smoke test: recover Qwen3-0.6B from durable Forge Git artifacts.

This intentionally uses NO prior Actions artifact or warm model cache. It loads
official Qwen base weights and the exact trained PEFT LoRA committed to Forge
main. Real learned per-attribute encoders send messages to real learned decoder.
Only the final evaluation compares against expected data; no oracle values
are passed from sender to receiver. Not evidence of general agent execution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "models/qwen3-0.6b-communication-lora-v2"
SYSTEM = "You exchange compact machine-readable messages. Reply with the exact required value and nothing else."
A = ["maple", "frost", "ocean", "cinder", "violet", "echo", "pollen", "tundra"]
B = ["marble", "orbit", "fable", "harbor", "spruce", "copper", "velvet", "comet"]
EXPECTED_SHA = "4ea165d25f8e4e11f8a306ddee3d27558043051699e969cd13ad251db6e8143f"
PAIRS = [(i, j) for i in range(8) for j in range(8) if (i+j) % 4 == 0]


def verify_local_checkpoint():
    weights = ADAPTER / "adapter_model.safetensors"
    config = ADAPTER / "adapter_config.json"
    manifest = json.loads((ADAPTER / "manifest.json").read_text())
    assert weights.is_file() and config.is_file(), "Missing durable Qwen adapter"
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    assert digest == EXPECTED_SHA == manifest["sha256_by_path"]["adapter_model.safetensors"]
    assert json.loads(config.read_text())["base_model_name_or_path"] == "Qwen/Qwen3-0.6B"
    assert len(PAIRS) == 16 and len(set(PAIRS)) == 16
    return digest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=4, help="pairs from the exact 16 pre-inspected heldout cases")
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--out", default="qwen_cold_start.json")
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()
    digest = verify_local_checkpoint()
    if args.self_test:
        print(f"CHECKPOINT VALID SHA256={digest} test_pairs={len(PAIRS)}", flush=True)
        return
    if not 1 <= args.limit <= len(PAIRS):
        p.error("--limit must be in [1,16]")

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    torch.set_num_threads(2)
    tok = AutoTokenizer.from_pretrained(args.model)
    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32,
                                               low_cpu_mem_usage=True).eval()
    net = PeftModel.from_pretrained(base, str(ADAPTER), is_trainable=False).eval()

    def ask(prompt):
        t = tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
            tokenize=True, add_generation_prompt=True,
            enable_thinking=False, return_tensors="pt")
        with torch.inference_mode():
            out = net.generate(t, attention_mask=torch.ones_like(t),
                               max_new_tokens=24, do_sample=False,
                               pad_token_id=tok.eos_token_id)
        return tok.decode(out[0, t.shape[1]:], skip_special_tokens=True).strip()

    records = []
    for i, j in PAIRS[:args.limit]:
        sx = ask(f"ROLE: ENCODE X\nVALUE: X={i}\nSYMBOL:")
        sy = ask(f"ROLE: ENCODE Y\nVALUE: Y={j}\nSYMBOL:")
        sent = sx + "|" + sy
        receiver = ask(f"ROLE: RECEIVE\nMESSAGE: {sent}\nOUTPUT X AND Y:")
        parsed = receiver.replace(" ", "").splitlines()[0] if receiver else ""
        passed = (sx == A[i] and sy == B[j] and parsed == f"X={i},Y={j}")
        record = dict(i=i, j=j, sender_x=sx, sender_y=sy,
                      actual_message=sent, actual_receiver=receiver,
                      expected_message=A[i]+"|"+B[j], correct=passed)
        records.append(record)
        print("COLD_START_HANDOFF", json.dumps(record), flush=True)
    result = dict(model=args.model, adapter_path=str(ADAPTER.relative_to(ROOT)),
                  adapter_sha256=digest, source="fresh Forge checkout, official pretrained base",
                  cases=records, correct=sum(int(r["correct"]) for r in records),
                  total=len(records), limitation="Inspected compositional game cases, not independent broad transfer")
    Path(args.out).write_text(json.dumps(result, indent=2))
    print("COLD_START_RESULT", result["correct"], "/", result["total"], flush=True)
    if result["correct"] != result["total"]:
        raise SystemExit("Qwen cold-start loaded but communication verification failed")


if __name__ == "__main__":
    main()
