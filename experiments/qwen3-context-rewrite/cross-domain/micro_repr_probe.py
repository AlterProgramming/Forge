#!/usr/bin/env python3
"""Frozen Qwen3 small-arithmetic reachability under changes of representation.

This is a diagnostic on arithmetic steps from ALREADY examined benchmark tasks.
It is not a blind cross-task score and no Qwen weights are updated.
For each step compare natural-language query, symbolic equation, and a Python
expression presented as text to the model; none runs Python to generate the
model's answer. Exact numeric evaluation is performed externally after inference.
"""
import argparse
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
import time
from boost import number

OPS=[
 ("elapsed","How many hours are there from 1 PM until 4 PM?","4 - 1","3"),
 ("distance","At 10 miles per hour for 3 hours, how many miles?","10 * 3","30"),
 ("daily_work","How many minutes per day if 1200 minutes are split across 5 days?","1200 / 5","240"),
 ("workday_minutes","How many minutes are in an 8 hour workday?","8 * 60","480"),
 ("two_discounts","What is the dollar value of 43 dollars minus 4 dollars minus 8 dollars?","43 - 4 - 8","31"),
 ("cereal_total","How many boxes in total are 3 boxes and 5 boxes?","3 + 5","8"),
 ("percentage","What percentage is 240 out of 480?","240 / 480 * 100","50"),
 ("reward_count","At $1 per $20, how many dollars reward on $80?","80 / 20","4"),
 ("coupon","What is twice a $4 coupon?","2 * 4","8"),
 ("difference","How much more is 8 than 4?","8 - 4","4")
]
def variants(natural,expr):
    return dict(
      natural=natural+"\nAnswer with a single numerical value.",
      symbolic=f"Evaluate this mathematical expression exactly:\n{expr} = ?\nOutput the number only.",
      python=f"What value does this Python program print?\nprint({expr})\nOutput only that number.",
      guided=f"Compute the arithmetic below without inventing other quantities. Break it into the necessary basic steps then report the final numeric answer:\n{expr}\nAnswer:"
    )
def equivalent(parsed,expected):
    if parsed is None:return False
    try:return Decimal(parsed)==Decimal(expected)
    except (InvalidOperation,ValueError):return False

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model",default="Qwen/Qwen3-0.6B")
    p.add_argument("--output",default="micro_repr_results.json")
    p.add_argument("--self-test",action="store_true")
    args=p.parse_args()
    if args.self_test:
        assert equivalent("1.2E+3","1200")
        assert not equivalent("1.2E+3","240")
        assert len(OPS)==10
        print("PASS numeric equivalence and fixed probe catalog")
        return
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    torch.set_num_threads(2)
    tok=AutoTokenizer.from_pretrained(args.model)
    model=AutoModelForCausalLM.from_pretrained(args.model,dtype=torch.float32,
                                               low_cpu_mem_usage=True).eval()
    device=next(model.parameters()).device
    rows=[]
    started=time.monotonic()
    def call(prompt):
        ids=tok.apply_chat_template([{"role":"user","content":prompt}],
            tokenize=True,add_generation_prompt=True,enable_thinking=False,
            return_tensors="pt").to(device)
        with torch.inference_mode():
            y=model.generate(ids,attention_mask=torch.ones_like(ids),do_sample=False,
                             max_new_tokens=80,pad_token_id=tok.eos_token_id)
        raw=tok.decode(y[0,ids.shape[1]:],skip_special_tokens=True).strip()
        return dict(prompt=prompt,raw=raw,parsed=number(raw),
                    input_tokens=int(ids.shape[1]),output_tokens=int(y.shape[1]-ids.shape[1]))
    for name,natural,expression,expected in OPS:
        arms={}
        for label,prompt in variants(natural,expression).items():
            a=call(prompt)
            a["correct"]=equivalent(a["parsed"],expected)
            arms[label]=a
            print(f"{name} {label}: expected={expected} got={a['parsed']} ok={a['correct']}",flush=True)
        rows.append(dict(name=name,expression=expression,expected=expected,arms=arms))
        counts={label:sum(int(r["arms"][label]["correct"]) for r in rows)
                for label in arms}
        summary=dict(total=len(rows),by_format=counts,
                     recovered_by_any=sum(int(any(a["correct"] for a in r["arms"].values()))
                                          for r in rows))
        Path(args.output).write_text(json.dumps(dict(model=args.model,config=vars(args),
            qualification="posthoc representation diagnostic; 10 micro operations from inspected cases",
            rows=rows,summary=summary,elapsed_seconds=round(time.monotonic()-started,2)),indent=2))
    print(json.dumps(summary,indent=2),flush=True)

if __name__=="__main__":main()
