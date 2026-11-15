#!/usr/bin/env python3
"""Game-specific Qwen decision -> learned message -> receiver, with immutable evidence.

The base model scores task-relevant alternatives; the *actual* recovered LoRA
encodes the selected route and a second LoRA call decodes it. The receiver's
output, not a gold label, determines the action sent downstream.
This is a bounded task-router, not a claim of novel game-trained weights.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from qwen_recovery_smoke import ADAPTER, EXPECTED_SHA, verify_local_checkpoint

ROOT = Path(__file__).resolve().parents[1]
GAME = ROOT / "deliveries/forge-arena/source/index.html"
SYSTEM = "You exchange compact machine-readable messages. Reply with the exact required value and nothing else."
CHOICES = (
    "0: add a visible Pause/Resume button for touch players and keep keyboard pause synchronized",
    "1: add an optional auto-fire button that uses the current weapon and cooldown rules",
    "2: add an accessible combat instructions disclosure near the battle controls",
    "3: add a costly emergency repair button usable during combat to spend energy on hull",
)
# Each pair is an exact independently inspectable trained SEND/RECEIVE path.
ROUTES = ((0, 0), (0, 4), (1, 3), (1, 7))
LABELS = ("touch-pause", "auto-fire", "combat-help", "energy-repair")

def digest(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def game_source() -> bytes:
    source = GAME.read_bytes()
    if not (b"<!doctype html>" in source.lower() and
            b"function begin()" in source and
            b"id=\"start\"" in source and b"function pulse()" in source):
        raise RuntimeError("Recovered Forge Arena source not present or not intact")
    return source

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--cc-gate-sha", default="")
    p.add_argument("--out", default="forge-arena-qwen-g1.json")
    args = p.parse_args()
    checkpoint = verify_local_checkpoint()
    source_sha = digest(game_source())
    assert checkpoint == EXPECTED_SHA
    assert len(set(ROUTES)) == len(ROUTES) == len(CHOICES)
    if args.self_test:
        print("G1_SOURCE_AND_LORA_VERIFIED", json.dumps({"source_sha256": source_sha, "adapter_sha256": checkpoint}), flush=True)
        return
    if len(args.cc_gate_sha) != 64 or any(c not in "0123456789abcdef" for c in args.cc_gate_sha):
        raise SystemExit("Valid Command Center G0 receipt SHA-256 is REQUIRED before Qwen execution")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.float32, low_cpu_mem_usage=True).eval()
    model = PeftModel.from_pretrained(base, str(ADAPTER), is_trainable=False).eval()

    context = (
        "You are the action router for a real deliverable, not a benchmark. "
        "The user provided an image of a mobile cartoon vehicle-combat upgrade game: "
        "Ancient Greece chapter, small three-slot blue truck, weapon store, coins, gems, "
        "waves of enemies and a large START battle control. "
        "The existing Forge Arena is already playable with weapons, energy, hull, upgrades, "
        "persistent coins and waves. Choose ONE concrete next improvement to that real source. "
        "Prefer immediate usability and preserving game mechanics.\n"
        + "\n".join(CHOICES)
        + "\nRespond with exactly one digit (0, 1, 2, or 3):"
    )
    tokenized = tokenizer.apply_chat_template(
        [{"role":"system","content":"Choose the most useful implementation action. No explanation."},
         {"role":"user","content":context}],
        tokenize=True, add_generation_prompt=True, enable_thinking=False, return_tensors="pt")
    with torch.inference_mode(), model.disable_adapter():
        logits = model(tokenized).logits[0, -1]
        logprobs = torch.log_softmax(logits.float(), dim=-1)
    ids = [tokenizer.encode(str(i), add_special_tokens=False) for i in range(4)]
    if any(len(x)!=1 for x in ids):
        raise RuntimeError("One-token candidate assumption does not hold")
    scores = [round(float(logprobs[x[0]].item()), 6) for x in ids]
    choice = max(range(4), key=lambda i: scores[i])
    # The game action is only derived from the receiver after the real LoRA handoff.
    i, j = ROUTES[choice]

    def ask_lora(prompt: str) -> str:
        ids = tokenizer.apply_chat_template(
            [{"role":"system","content":SYSTEM},{"role":"user","content":prompt}],
            tokenize=True, add_generation_prompt=True, enable_thinking=False, return_tensors="pt")
        with torch.inference_mode():
            out = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=24,
                                 do_sample=False, pad_token_id=tokenizer.eos_token_id)
        return tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()

    sx = ask_lora(f"ROLE: ENCODE X\nVALUE: X={i}\nSYMBOL:")
    sy = ask_lora(f"ROLE: ENCODE Y\nVALUE: Y={j}\nSYMBOL:")
    message = sx + "|" + sy
    receiver = ask_lora(f"ROLE: RECEIVE\nMESSAGE: {message}\nOUTPUT X AND Y:")
    parsed = receiver.replace(" ", "").splitlines()[0] if receiver else ""
    decoded = None
    for k, pair in enumerate(ROUTES):
        if parsed == f"X={pair[0]},Y={pair[1]}":
            decoded = k
            break
    valid = decoded == choice and sx and sy
    result = {
        "gate": "G1-real-Qwen-game-intent-and-communication",
        "status": "PASS" if valid else "FAIL",
        "command_center_g0_receipt_sha256": args.cc_gate_sha,
        "model": "Qwen/Qwen3-0.6B",
        "adapter": str(ADAPTER.relative_to(ROOT)),
        "adapter_sha256": checkpoint,
        "game_source_sha256": source_sha,
        "actual_user_goal": "Improve and deliver the provided Ancient Greece truck combat game through existing BrightEngine system",
        "options": list(CHOICES),
        "decision": {"method":"base Qwen next-token restricted logprob choice; adapter disabled for selection",
                     "scores":scores, "selected_before_handoff":choice},
        "learned_handoff": {"sender_x":sx,"sender_y":sy,"message":message,"receiver":receiver,
                            "decoded_action":decoded,"passes":bool(valid)},
        "next_action": {"id":decoded,"label":LABELS[decoded], "description":CHOICES[decoded]} if valid else None,
        "limitations": "Four already inspected protocol pairs, not a new blind generalization benchmark; task choice is bounded to approved operations.",
    }
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print("G1_QWEN_TASK_ROUTE", json.dumps(result), flush=True)
    if not valid:
        raise SystemExit("FAIL CLOSED: actual trained sender->receiver failed; no gold-route substitution")

if __name__ == "__main__":
    main()
