#!/usr/bin/env python3
"""Test-time Qwen3 improvement on independent published tasks.

Each task gets K samples. Baseline = the FIRST sample. Boosted = plurality
for math/MCQ, or the first sample passing a SINGLE development test for code.
All outcomes are reported on labels/held-out code tests not used for selection.
No claims about fine-tuned weights, free compute, or guaranteed transfer.
"""
import argparse
import ast
from collections import Counter
from decimal import Decimal, InvalidOperation
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

SYSTEM = "Follow the requested response format exactly. Be concise."
NUM = re.compile(r"[-+]?(?:\d[\d,]*)(?:\.\d+)?")
FINAL_NUM = re.compile(r"(?:final\s*answer|answer)\s*[:=]\s*([$]?\s*[-+]?(?:\d[\d,]*)(?:\.\d+)?)", re.I)
FINAL_CHOICE = re.compile(r"(?:final\s*answer|answer)\s*[:=]\s*([A-E]|[1-5])\b", re.I)

def number(text):
    hits = FINAL_NUM.findall(text)
    if not hits:
        hits = NUM.findall(text)
    if not hits:
        return None
    try:
        return str(Decimal(hits[-1].replace(",", "").replace("$", "").strip()).normalize())
    except InvalidOperation:
        return None

def choice(text, labels):
    hits = FINAL_CHOICE.findall(text.upper())
    if not hits:
        hits = re.findall(r"\b[A-E]\b", text.upper())
    if not hits:
        return None
    candidate = hits[-1]
    if candidate in labels:
        return candidate
    if candidate.isdigit() and 1 <= int(candidate) <= len(labels):
        return labels[int(candidate) - 1]
    return None

def python_code(text):
    fence = chr(96) * 3
    if fence in text:
        match = re.search(re.escape(fence) + r"(?:python)?\s*\n(.*?)" + re.escape(fence), text, re.S | re.I)
        if match:
            return match.group(1).strip()
    return text.strip()

def run_tests(code, tests, setup="", timeout=3):
    """For public MBPP tests only. Run exclusively on disposable isolated workers.

    This is a timeout/resource-limited subprocess, NOT a security sandbox.
    Never run untrusted generated code on a personal machine.
    """
    if not tests:
        return False
    try:
        ast.parse(code)
    except (SyntaxError, ValueError):
        return False
    payload = (
        "import resource\n"
        "resource.setrlimit(resource.RLIMIT_CPU, (2, 2))\n"
        "resource.setrlimit(resource.RLIMIT_AS, (512*1024*1024, 512*1024*1024))\n"
        + setup + "\n" + code + "\n" + "\n".join(tests) + "\n"
    )
    with tempfile.TemporaryDirectory() as folder:
        try:
            result = subprocess.run(
                [sys.executable, "-I", "-c", payload],
                cwd=folder, stdin=subprocess.DEVNULL, capture_output=True,
                timeout=timeout, env={"PYTHONHASHSEED": "0"},
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, OSError):
            return False

def load_tasks(n, seed):
    from datasets import load_dataset
    rng = random.Random(seed)
    def sample(ds):
        idx = rng.sample(range(len(ds)), min(n, len(ds)))
        return [ds[i] for i in idx]
    math_ds = sample(load_dataset("openai/gsm8k", "main", split="test"))
    arc_ds = sample(load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test"))
    code_ds = sample(load_dataset("google-research-datasets/mbpp", "full", split="test"))
    tasks = []
    for i, ex in enumerate(math_ds):
        # Evaluation-only labels are NOT passed to prompting/selection.
        gold = number(ex["answer"].split("####")[-1])
        if gold is not None:
            tasks.append(dict(family="gsm8k", id=str(i),
                              prompt=ex["question"] + "\nGive your final result as: FINAL ANSWER: <number>",
                              gold=gold))
    for ex in arc_ds:
        labels = [str(s) for s in ex["choices"]["label"]]
        options = "\n".join(f"{lab}. {txt}" for lab, txt in zip(labels, ex["choices"]["text"]))
        tasks.append(dict(family="arc", id=ex["id"],
                          prompt=ex["question"] + "\n" + options
                          + "\nOutput a single choice as: FINAL ANSWER: <letter>",
                          gold=str(ex["answerKey"]), labels=labels))
    for ex in code_ds:
        tests = list(ex["test_list"])
        if len(tests) < 2:
            continue
        # Reveal only a function name, not the expected outputs/test cases.
        match = re.search(r"\bassert\s+([a-zA-Z_]\w*)\s*\(", tests[0])
        signature = f"\nUse the function name {match.group(1)}." if match else ""
        tasks.append(dict(family="mbpp", id=str(ex["task_id"]),
                          prompt=ex["text"] + signature
                          + "\nReturn only runnable Python code, no explanation.",
                          dev_tests=tests[:1], eval_tests=tests[1:],
                          setup=ex.get("test_setup_code") or ""))
    return tasks

def vote(values):
    valid = [x for x in values if x is not None]
    if not valid:
        return None, -1
    counts = Counter(valid)
    top = max(counts.values())
    for i, value in enumerate(values):
        if value is not None and counts[value] == top:
            return value, i
    return None, -1

def decide(task, responses):
    family = task["family"]
    if family == "gsm8k":
        parsed = [number(s) for s in responses]
        chosen, index = vote(parsed)
    elif family == "arc":
        parsed = [choice(s, task["labels"]) for s in responses]
        chosen, index = vote(parsed)
    else:
        parsed = [python_code(s) for s in responses]
        accepted = [run_tests(code, task["dev_tests"], task["setup"]) for code in parsed]
        index = next((i for i, passed in enumerate(accepted) if passed), 0)
        chosen = parsed[index]
    return parsed, chosen, index

def correct(task, candidate):
    if candidate is None:
        return False
    if task["family"] == "mbpp":
        return run_tests(candidate, task["eval_tests"], task["setup"])
    return candidate == task["gold"]

def summarize(rows):
    groups = {}
    for family in ("gsm8k", "arc", "mbpp"):
        data = [x for x in rows if x["family"] == family]
        if data:
            groups[family] = dict(n=len(data), baseline=sum(x["baseline_correct"] for x in data),
                                  boosted=sum(x["boosted_correct"] for x in data))
    groups["all"] = dict(n=len(rows), baseline=sum(x["baseline_correct"] for x in rows),
                         boosted=sum(x["boosted_correct"] for x in rows))
    return groups

def self_test():
    assert number("FINAL ANSWER: 1,205.50") == "1205.5"
    assert number("compute 10 then FINAL ANSWER: -2") == "-2"
    assert choice("FINAL ANSWER: C", ["A","B","C","D"]) == "C"
    assert python_code(chr(96)*3 + "python\nx=1\n" + chr(96)*3) == "x=1"
    assert vote(["A","B","B"])[0] == "B"
    ex = dict(family="mbpp", dev_tests=["assert f(2)==4"],
              eval_tests=["assert f(3)==6"], setup="")
    parsed, selected, i = decide(ex, ["def f(x): return x+1","def f(x): return x*2"])
    assert i == 1 and correct(ex, selected) and not correct(ex, parsed[0])
    assert not run_tests("def f(x): return x*2", ["assert f(3)==7"])
    print("PASS: parsing, independent verification, and first-pass selection", flush=True)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--n", type=int, default=4, help="fixed-seed samples per family")
    p.add_argument("--k", type=int, default=3, help="candidates per task")
    p.add_argument("--seed", type=int, default=20261114)
    p.add_argument("--max-tokens", type=int, default=160)
    p.add_argument("--output", default="cross_domain_results.json")
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()
    if args.self_test:
        self_test()
        return
    if args.n < 1 or args.k < 1:
        p.error("--n and --k must be positive")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(2)
    tasks = load_tasks(args.n, args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32,
                                                  low_cpu_mem_usage=True).eval()
    device = next(model.parameters()).device
    rows = []
    results = dict(config=vars(args), model=args.model,
                   protocol="baseline=first_sample; boost=plurality or first passing dev test",
                   records=rows)
    start = time.monotonic()
    for ti, task in enumerate(tasks):
        responses = []
        input_tokens = generated_tokens = 0
        for ki in range(args.k):
            torch.manual_seed(args.seed + 104729*ti + ki)
            messages = [{"role": "system", "content": SYSTEM},
                        {"role": "user", "content": task["prompt"]}]
            inputs = tokenizer.apply_chat_template(messages, tokenize=True,
                      add_generation_prompt=True, enable_thinking=False,
                      return_tensors="pt").to(device)
            with torch.inference_mode():
                output = model.generate(inputs, do_sample=True, temperature=0.7,
                                        top_p=0.8, top_k=20,
                                        max_new_tokens=args.max_tokens,
                                        pad_token_id=tokenizer.eos_token_id)
            responses.append(tokenizer.decode(output[0, inputs.shape[1]:],
                                              skip_special_tokens=True))
            input_tokens += int(inputs.shape[1])
            generated_tokens += int(output.shape[1] - inputs.shape[1])
        parsed, chosen, index = decide(task, responses)
        baseline = parsed[0]
        record = dict(family=task["family"], id=task["id"], prompt=task["prompt"],
                      baseline_correct=correct(task, baseline),
                      boosted_correct=correct(task, chosen), chosen_index=index,
                      candidates=responses, parsed=parsed, selected=chosen,
                      input_tokens=input_tokens, generated_tokens=generated_tokens)
        # Do not publish hidden label or MBPP held-out tests in prompt.
        rows.append(record)
        results["summary"] = summarize(rows)
        results["elapsed_seconds"] = round(time.monotonic()-start, 2)
        Path(args.output).write_text(json.dumps(results, indent=2))
        print(f'{len(rows)}/{len(tasks)} {task["family"]}: baseline={record["baseline_correct"]} boosted={record["boosted_correct"]}', flush=True)
    print(json.dumps(results["summary"], indent=2), flush=True)

if __name__ == "__main__":
    main()
