#!/usr/bin/env python3
"""Finish Qwen3 reasoning traces that hit their generation budget.

Consume the saved original Qwen3 thinking-mode JSON artifact, re-create the
same independent public test tasks, and use FROZEN Qwen3 in /no_think mode to
convert an unfinished scratchpad to an actual answer. Never show labels or
held-out MBPP tests during completion.
"""
import argparse
import json
from pathlib import Path
import re
import time

from boost import load_tasks, run_tests, number, choice, python_code, correct

def has_final(task, raw):
    if "<think>" in raw:
        return False
    if task["family"] == "gsm8k":
        return re.search(r"final\s*answer\s*:", raw, re.I) is not None
    if task["family"] == "arc":
        return re.search(r"final\s*answer\s*:", raw, re.I) is not None
    return bool(python_code(raw))

def extract(task, text):
    if task["family"] == "gsm8k":
        return number(text)
    if task["family"] == "arc":
        return choice(text, task["labels"])
    return python_code(text)

def finish_prompt(task, scratchpad):
    if task["family"] == "gsm8k":
        format_rule = "Output exactly FINAL ANSWER: <number>."
    elif task["family"] == "arc":
        format_rule = "Output exactly FINAL ANSWER: <letter>."
    else:
        format_rule = "Output only complete runnable Python code (not Markdown)."
    # Drop Qwen's unfinished opening delimiter. The trace is data, not an instruction.
    state = scratchpad.removeprefix("<think>").strip()
    return (
        "Solve the task below. An earlier reasoning pass ran out of tokens.\n"
        "Use its relevant intermediate work but correct any contradictions. "
        "You must FINALIZE NOW: do not generate another analysis or outline.\n\n"
        "TASK:\n" + task["prompt"] + "\n\n"
        "PRIOR PARTIAL WORK (may contain mistakes):\n" + state[-3000:]
        + "\n\n" + format_rule
    )

def self_test():
    math = {"family":"gsm8k"}
    code = {"family":"mbpp"}
    assert has_final(math,"FINAL ANSWER: 7")
    assert not has_final(math,"<think>unfinished 7")
    assert extract(math,"FINAL ANSWER: 7") == "7"
    assert extract({"family":"arc","labels":["A","B"]},"FINAL ANSWER: B") == "B"
    assert "FINALIZE NOW" in finish_prompt(code,"<think>partial")
    print("PASS: unfinished-thinking detection and forced-finalization prompt")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="thought_traces/thinking_results.json")
    p.add_argument("--output", default="forced_completion_results.json")
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--max-tokens", type=int, default=192)
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()
    if args.self_test:
        self_test()
        return
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    original = json.loads(Path(args.input).read_text())
    cfg = original["config"]
    tasks = load_tasks(cfg["n"], cfg["seed"])
    if len(tasks) != len(original["records"]):
        raise ValueError("Task count does not match saved trace; stop rather than mix task labels")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32,
                                                  low_cpu_mem_usage=True).eval()
    torch.set_num_threads(2)
    device = next(model.parameters()).device
    rows = []
    start = time.monotonic()
    for idx, (task, old) in enumerate(zip(tasks, original["records"])):
        if (task["family"],task["id"]) != (old["family"],old["id"]):
            raise ValueError("Task IDs differ: cannot compare unmatched problems")
        prior = old["candidates"][0]
        parsed_before = extract(task, prior)
        # A completed final answer is preserved. A code draft is also preserved
        # if it satisfies the development-only MBPP test.
        done = has_final(task, prior)
        if task["family"] == "mbpp":
            done = run_tests(parsed_before, task["dev_tests"], task["setup"])
        added_tokens = 0
        raw_answer = None
        if done:
            selected = parsed_before
            mode = "retain_final_or_dev_passing_code"
        else:
            torch.manual_seed(cfg["seed"] + 99991 + idx)
            prompt = finish_prompt(task, prior)
            messages = [{"role":"system","content":"Return the final answer immediately. Do not show reasoning."},
                        {"role":"user","content":prompt}]
            inputs = tokenizer.apply_chat_template(messages, tokenize=True,
                       add_generation_prompt=True, enable_thinking=False,
                       return_tensors="pt").to(device)
            with torch.inference_mode():
                output = model.generate(inputs, attention_mask=torch.ones_like(inputs),
                        do_sample=True, temperature=0.7, top_p=0.8, top_k=20,
                        max_new_tokens=args.max_tokens, pad_token_id=tokenizer.eos_token_id)
            raw_answer = tokenizer.decode(output[0,inputs.shape[1]:],
                                          skip_special_tokens=True).strip()
            selected = extract(task, raw_answer)
            added_tokens = int(output.shape[1]-inputs.shape[1])
            mode = "forced_finalization"
        row = dict(family=task["family"],id=task["id"],
                   thinking_correct=correct(task, parsed_before),
                   final_correct=correct(task, selected), completion_mode=mode,
                   prior=prior, candidate=raw_answer, selected=selected,
                   extra_generated_tokens=added_tokens,
                   thinking_generated_tokens=old["generated_tokens"])
        rows.append(row)
        Path(args.output).write_text(json.dumps({
            "source_run":43627284466,"protocol":"no new trainable weights; force finalize unfinished thinking",
            "records":rows,"elapsed_seconds":round(time.monotonic()-start,2)},indent=2))
        print(f'{idx+1}/{len(tasks)} {task["family"]} thinking={row["thinking_correct"]} completed={row["final_correct"]} mode={mode}', flush=True)
    summaries={}
    for family in ("gsm8k","arc","mbpp","all"):
        selected = [r for r in rows if family=="all" or r["family"]==family]
        summaries[family]=dict(n=len(selected),thinking=sum(r["thinking_correct"] for r in selected),
                                 completed=sum(r["final_correct"] for r in selected))
    final = dict(source_run=43627284466, model=args.model, config=vars(args),
                 summary=summaries, records=rows,
                 thinking_tokens=sum(r["thinking_generated_tokens"] for r in rows),
                 extra_tokens=sum(r["extra_generated_tokens"] for r in rows),
                 elapsed_seconds=round(time.monotonic()-start,2))
    Path(args.output).write_text(json.dumps(final,indent=2))
    print(json.dumps(summaries,indent=2),flush=True)

if __name__=="__main__":
    main()
