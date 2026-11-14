#!/usr/bin/env python3
"""Exact causal A/B: same Qwen3 arithmetic question, only output instruction changes.

Twelve existing arithmetic micro-questions from the four analyzed failures.
Frozen Qwen3-0.6B, deterministic decoding, *same token limits*, questions,
tokenization pipeline, model weights, and scoring. Only suffix is changed.

This isolates whether answer-format conditioning is a bottleneck. Posthoc
diagnostic on inspected cases, not independent transfer evidence.
"""
import argparse
import json
import time
from pathlib import Path
from decimal import Decimal, InvalidOperation
from trace_probe import CASES
from boost import number

SUFFIXES={
  "original_format":"Return only FINAL ANSWER: <number>, with no explanation.",
  "no_placeholder":"Return only FINAL ANSWER: followed by the numeric value, without any angle brackets or explanation.",
  "direct_numeric":"Answer with a single numerical value."
}

def equivalent(pred,truth):
    if pred is None: return False
    try: return Decimal(pred)==Decimal(truth)
    except (ValueError,InvalidOperation): return False

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model",default="Qwen/Qwen3-0.6B")
    p.add_argument("--output",default="format_causal_results.json")
    p.add_argument("--self-test",action="store_true")
    args=p.parse_args()
    if args.self_test:
        assert equivalent("1.2E+3","1200")
        assert len(SUFFIXES)==3
        assert sum(len(c["micro"]) for c in CASES.values())==12
        print("PASS exact question catalog and Decimal scoring",flush=True)
        return
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    torch.set_num_threads(2)
    tok=AutoTokenizer.from_pretrained(args.model)
    model=AutoModelForCausalLM.from_pretrained(args.model,dtype=torch.float32,
                                               low_cpu_mem_usage=True).eval()
    device=next(model.parameters()).device
    started=time.monotonic()
    rows=[]
    def ask(prompt):
        ids=tok.apply_chat_template([{"role":"user","content":prompt}],
            tokenize=True,add_generation_prompt=True,enable_thinking=False,
            return_tensors="pt").to(device)
        with torch.inference_mode():
            y=model.generate(ids,attention_mask=torch.ones_like(ids),do_sample=False,
                             max_new_tokens=48,pad_token_id=tok.eos_token_id)
        raw=tok.decode(y[0,ids.shape[1]:],skip_special_tokens=True).strip()
        return dict(raw=raw,parsed=number(raw),input_tokens=int(ids.shape[1]),
                    output_tokens=int(y.shape[1]-ids.shape[1]),prompt=prompt)
    for case_id,case in CASES.items():
        for qnum,(question,expected) in enumerate(case["micro"],1):
            arms={}
            for label,suffix in SUFFIXES.items():
                observed=ask(question+"\n"+suffix)
                observed["correct"]=equivalent(observed["parsed"],expected)
                arms[label]=observed
                print(f'{case_id}.{qnum} {label}: expected={expected} got={observed["parsed"]} correct={observed["correct"]}',flush=True)
            rows.append(dict(case=case_id,stage=qnum,question=question,
                             expected=expected,arms=arms))
            result=dict(qualification="posthoc within-question output-format intervention",
                        model=args.model,config=vars(args),rows=rows,
                        summary={label:sum(int(r["arms"][label]["correct"]) for r in rows)
                                 for label in SUFFIXES},
                        elapsed_seconds=round(time.monotonic()-started,2))
            Path(args.output).write_text(json.dumps(result,indent=2))
    print(json.dumps(result["summary"],indent=2),flush=True)

if __name__=="__main__":main()
