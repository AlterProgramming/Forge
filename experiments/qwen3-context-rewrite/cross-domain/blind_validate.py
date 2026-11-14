#!/usr/bin/env python3
"""Preregistered cross-domain replication on unexamined benchmark questions.

Frozen Qwen3 0.6B; exact three protocols from earlier exploration:
  (1) single nonthinking candidate (160 tokens),
  (2) single thinking candidate (512 tokens),
  (3) same thinking candidate + forced completion if unfinished (192 tokens).
Selection cannot access gold answers or held-out MBPP tests.
Prior inspected prompts are excluded BEFORE Qwen inference.
"""
import argparse
import json
import random
import time
from pathlib import Path

from boost import load_tasks, run_tests, correct
from force_complete import extract, has_final, finish_prompt

def choose_fresh(family, count, seed, prior_json):
    inspected = json.loads(Path(prior_json).read_text())
    excluded = {r["prompt"] for r in inspected["records"]}
    # Sample extra examples in predetermined order to replace any overlap.
    pool = load_tasks(count + 32, seed)
    selected = [t for t in pool if t["family"] == family and t["prompt"] not in excluded][:count]
    if len(selected) != count:
        raise RuntimeError("Not enough non-overlapping tasks to evaluate")
    if len({t["prompt"] for t in selected}) != len(selected):
        raise RuntimeError("Duplicate tasks in selected evaluation")
    return selected

def generate(model, tokenizer, device, messages, *, thinking, max_tokens, seed):
    import torch
    torch.manual_seed(seed)
    tokens = tokenizer.apply_chat_template(messages, tokenize=True,
               add_generation_prompt=True, enable_thinking=thinking,
               return_tensors="pt").to(device)
    with torch.inference_mode():
        full = model.generate(tokens,attention_mask=torch.ones_like(tokens),
               do_sample=True,temperature=0.6 if thinking else 0.7,
               top_p=0.95 if thinking else 0.8,top_k=20,
               max_new_tokens=max_tokens,pad_token_id=tokenizer.eos_token_id)
    raw = tokenizer.decode(full[0,tokens.shape[1]:],skip_special_tokens=False)
    if thinking and "</think>" in raw:
        raw = raw.split("</think>", 1)[-1]
    return raw.replace("<|im_end|>","").strip(), int(tokens.shape[1]), int(full.shape[1]-tokens.shape[1])

def self_test():
    example = {"family":"gsm8k","prompt":"Add 2 and 3."}
    assert not has_final(example,"<think>2+3")
    assert "FINALIZE NOW" in finish_prompt(example,"<think>2+3")
    print("PASS: replication invariants; labels remain evaluation-only")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--family",choices=["gsm8k","arc","mbpp"],required=False)
    p.add_argument("--n",type=int,default=6)
    p.add_argument("--seed",type=int,default=20261115)
    p.add_argument("--prior",default="prior_traces/cross_domain_results.json")
    p.add_argument("--model",default="Qwen/Qwen3-0.6B")
    p.add_argument("--output",default="blind_result.json")
    p.add_argument("--self-test",action="store_true")
    args=p.parse_args()
    if args.self_test:
        self_test()
        return
    if args.family is None or args.n < 1:
        p.error("--family is required, and --n must be positive")
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    torch.set_num_threads(2)
    tasks=choose_fresh(args.family,args.n,args.seed,args.prior)
    tok=AutoTokenizer.from_pretrained(args.model)
    model=AutoModelForCausalLM.from_pretrained(args.model,torch_dtype=torch.float32,
                                              low_cpu_mem_usage=True).eval()
    device=next(model.parameters()).device
    rows=[]
    start=time.monotonic()
    for i, task in enumerate(tasks):
        messages=[{"role":"system","content":"Follow the requested output format."},
                  {"role":"user","content":task["prompt"]}]
        plain,pi,po=generate(model,tok,device,messages,thinking=False,max_tokens=160,
                             seed=args.seed + 100003*i + 1)
        thinking,ti,to=generate(model,tok,device,messages,thinking=True,max_tokens=512,
                                seed=args.seed + 100003*i + 2)
        original=extract(task,thinking)
        keep=has_final(task,thinking)
        if task["family"]=="mbpp":
            keep=run_tests(original,task["dev_tests"],task["setup"])
        extra_input=extra_output=0
        if keep:
            completed_raw=thinking
            chosen=original
        else:
            completed_raw,extra_input,extra_output=generate(
                model,tok,device,
                [{"role":"system","content":"Return the final answer immediately. Do not show reasoning."},
                 {"role":"user","content":finish_prompt(task,thinking)}],
                thinking=False,max_tokens=192,seed=args.seed + 100003*i + 3)
            chosen=extract(task,completed_raw)
        plain_answer=extract(task,plain)
        row=dict(family=task["family"],id=task["id"],prompt=task["prompt"],
                 plain_correct=correct(task,plain_answer),
                 thinking_correct=correct(task,original),
                 completed_correct=correct(task,chosen),
                 plain_raw=plain,thinking_raw=thinking,completion_raw=completed_raw,
                 plain_answer=plain_answer,thinking_answer=original,
                 completed_answer=chosen,kept=keep,
                 generated_tokens_plain=po,generated_tokens_thinking=to,
                 generated_tokens_finish=extra_output,
                 input_tokens_total=pi+ti+extra_input)
        rows.append(row)
        counts=lambda key:sum(int(x[key]) for x in rows)
        result=dict(protocol="fixed; no new adaptation after 12-question exploratory trial",
                    config=vars(args),family=args.family,source_seed=args.seed,
                    rows=rows, summary=dict(n=len(rows),plain=counts("plain_correct"),
                    thinking=counts("thinking_correct"),completed=counts("completed_correct")),
                    tokens=dict(plain=sum(x["generated_tokens_plain"] for x in rows),
                    thinking=sum(x["generated_tokens_thinking"] for x in rows),
                    completion=sum(x["generated_tokens_finish"] for x in rows)),
                    elapsed_seconds=round(time.monotonic()-start,2))
        Path(args.output).write_text(json.dumps(result,indent=2))
        print(f'{i+1}/{len(tasks)} {args.family}: plain={row["plain_correct"]} think={row["thinking_correct"]} finish={row["completed_correct"]}',flush=True)
    print(json.dumps(result["summary"],indent=2),flush=True)
    print(json.dumps(result["tokens"],indent=2),flush=True)

if __name__=="__main__":
    main()
