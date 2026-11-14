#!/usr/bin/env python3
"""Run four researcher-supplied arithmetic dependency chains through frozen Qwen.

Each next question uses prior Qwen output, NOT an oracle answer. This tests
whether the computation is reachable when intermediate results persist and only
one operation is required per short context. The plan is supplied by the
researcher, so this is NOT an autonomous Qwen improvement or blind benchmark.
"""
import argparse
from decimal import Decimal
import json
from pathlib import Path
import time
from trace_probe import CASES, pick_tasks
from boost import number

def equal(a,b):
    if a is None:return False
    try:return Decimal(str(a))==Decimal(str(b))
    except Exception:return False

def steps(k,vals):
    if k=="1":
        return [
            lambda:"How many hours are there from 1 PM until 4 PM?",
            lambda:f"At 10 miles per hour for {vals[0]} hours, how many miles?",
            lambda:f"At 6 miles per hour, how many hours for {vals[1]} miles?"
        ]
    if k=="3":
        return [
            lambda:"How many minutes in total to clean 80 classrooms at 15 minutes each?",
            lambda:f"How many minutes per day if {vals[0]} minutes are divided evenly across 5 days?",
            lambda:"How many minutes are in an 8-hour workday?",
            lambda:f"What percentage is {vals[1]} out of {vals[2]}?"
        ]
    if k=="4":
        return [
            lambda:"At 1 dollar reward per 20 dollars spent, how many reward dollars from 80 dollars?",
            lambda:f"If a coupon is twice a {vals[0]} dollar reward, how many dollars is that coupon?",
            lambda:f"What is 43 minus {vals[0]} minus {vals[1]}?"
        ]
    if k=="5":
        return [
            lambda:"How many cereal boxes in total are 3 boxes and 5 boxes?",
            lambda:f"How much more is {vals[0]} than 4?"
        ]
    raise ValueError(k)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model",default="Qwen/Qwen3-0.6B")
    p.add_argument("--output",default="chained_computation_results.json")
    p.add_argument("--self-test",action="store_true")
    args=p.parse_args()
    if args.self_test:
        assert len(steps("3",[]))==4
        assert len(steps("5",[]))==2
        assert equal("5E+1","50")
        print("PASS chain shapes and numeric scorer")
        return
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    torch.set_num_threads(2)
    chosen=pick_tasks(20261115)
    tok=AutoTokenizer.from_pretrained(args.model)
    model=AutoModelForCausalLM.from_pretrained(args.model,dtype=torch.float32,low_cpu_mem_usage=True).eval()
    device=next(model.parameters()).device
    def ask(question):
        prompt=question+"\nAnswer with a single numerical value."
        ids=tok.apply_chat_template([{"role":"user","content":prompt}],tokenize=True,
                 add_generation_prompt=True,enable_thinking=False,return_tensors="pt").to(device)
        with torch.inference_mode():
            y=model.generate(ids,attention_mask=torch.ones_like(ids),do_sample=False,
                             max_new_tokens=80,pad_token_id=tok.eos_token_id)
        raw=tok.decode(y[0,ids.shape[1]:],skip_special_tokens=True).strip()
        return dict(prompt=prompt,raw=raw,parsed=number(raw),
                    input_tokens=int(ids.shape[1]),output_tokens=int(y.shape[1]-ids.shape[1]))
    rows=[]
    start=time.monotonic()
    for k,case in CASES.items():
        direct=ask(chosen[k]["prompt"])
        own=[]
        trace=[]
        for i,make_question in enumerate(steps(k,own)):
            if any(v is None for v in own):
                own.append(None)
                trace.append(dict(step=i+1,blocked=True,correct=False))
                continue
            question=make_question()
            response=ask(question)
            response["expected"]=case["micro"][i][1]
            response["correct"]=equal(response["parsed"],response["expected"])
            # IMPORTANT: numeric parser returns scientific notation after
            # Decimal.normalize (e.g. 30 -> '3E+1'). Preserve numerical
            # meaning and use plain decimal notation at state boundaries.
            parsed=response["parsed"]
            passed_to_next=format(Decimal(parsed),"f") if parsed is not None else None
            response["passed_to_next"]=passed_to_next
            trace.append(response)
            own.append(passed_to_next)
            print(f'{k} step {i+1}: want={response["expected"]} produced={response["parsed"]} correct={response["correct"]}',flush=True)
        row=dict(case=k,name=case["name"],gold=case["gold"],
                 direct=direct,direct_correct=equal(direct["parsed"],case["gold"]),
                 chained_output=own[-1],chained_correct=equal(own[-1],case["gold"]),
                 trace=trace,all_nodes_correct=all(t.get("correct") for t in trace))
        rows.append(row)
        report=dict(model=args.model,config=vars(args),protocol=
                    "hand-supplied decomposition; model-written values passed forward without correction; previously inspected problems",
                    rows=rows,summary=dict(n=len(rows),direct=sum(r["direct_correct"] for r in rows),
                    chained=sum(r["chained_correct"] for r in rows),
                    all_nodes_correct=sum(r["all_nodes_correct"] for r in rows)),
                    elapsed_seconds=round(time.monotonic()-start,2))
        Path(args.output).write_text(json.dumps(report,indent=2))
        print(f'{k} direct={row["direct_correct"]} chained={row["chained_correct"]}',flush=True)
    print(json.dumps(report["summary"],indent=2),flush=True)

if __name__=="__main__":main()
