#!/usr/bin/env python3
"""Causal localization of Qwen3-0.6B multi-fact reasoning failures.

Four GSM8K test examples are FIXED by their IDs from the *already inspected*
blind evaluation run 43629177966. This is a DIAGNOSTIC of reachability and
prompt state, not held-out performance evidence.

Control: original prompt.
Interventions: generic reminder, supplied valid intermediate facts,
supplied facts and relation/formula (but no final number).
Microtraces: independently query each arithmetic dependency as a small
single-step problem, verifying each intermediate against an oracle.
Same pretrained Qwen weights, deterministic decoding and fixed max-token
budget for every full-question arm.

A successful fact/bridge intervention establishes local prompt-based
recoverability, not a learned cross-task capability or a causal localization
inside transformer neurons. Every prompt and answer is preserved.
"""
import argparse
from collections import Counter
from decimal import Decimal
import json
import time
from pathlib import Path
from boost import load_tasks, number

# Entries were hand-diagnosed on PREVIOUSLY inspected failures. Numerical facts
# are independently checkable consequences of each word problem; the final
# answer is deliberately absent from the intervention prompts.
CASES = {
  "1": {
    "name":"Tom outbound duration -> distance -> return duration",
    "expect_terms":["Tom's ship", "1 to 4 PM"],
    "facts": [
      "Outbound sailing is from 1 PM until 4 PM: a duration of 3 hours.",
      "Outbound speed is 10 miles/hour. Return speed is 6 miles/hour."
    ],
    "bridge": [
      "The return covers the SAME DISTANCE as the outward trip.",
      "outward miles = outward speed * outward hours.",
      "return hours = outward miles / return speed."
    ],
    "micro":[
      ["How many hours elapse from 1 PM to 4 PM?", "3"],
      ["At 10 miles/hour for 3 hours, how many miles are traveled?", "30"],
      ["At 6 miles/hour, how many hours are needed for 30 miles?", "5"]
    ],
    "gold":"5"
  },
  "3": {
    "name":"Custodian total -> per-day -> fraction of daily capacity",
    "expect_terms":["80 classrooms","5 days"],
    "facts":[
      "Total cleaning is 80 * 15 = 1200 minutes for ALL FIVE DAYS, not per day.",
      "The work is divided evenly across 5 days.",
      "Each 8-hour day contains 480 minutes."
    ],
    "bridge":[
      "Daily cleaning minutes = total cleaning minutes / number of days.",
      "Daily cleaning percentage = daily cleaning minutes / 480 * 100."
    ],
    "micro":[
      ["How many minutes are required for 80 classrooms at 15 minutes each?", "1200"],
      ["If 1200 minutes of work are spread evenly across 5 days, how many minutes per day?", "240"],
      ["How many minutes are in an 8-hour workday?", "480"],
      ["What percent is 240 minutes out of 480 minutes?", "50"]
    ],
    "gold":"50"
  },
  "4": {
    "name":"Loyalty reward AND coupon, not one or the other",
    "expect_terms":["loyalty card","coupon"],
    "facts":[
      "The last purchase earned 80 / 20 = 4 dollars of reward credit.",
      "The additional coupon reduces the price by 2 * 4 = 8 dollars.",
      "The current subtotal is 43 dollars."
    ],
    "bridge":[
      "The reward credit AND the additional coupon both apply to this purchase.",
      "Amount paid = current subtotal - reward credit - coupon discount."
    ],
    "micro":[
      ["If a customer gets $1 per $20 spent, how many dollars of credit from $80?", "4"],
      ["If coupon discount is twice a $4 reward credit, what is the coupon discount in dollars?", "8"],
      ["What is 43 minus 4 minus 8?", "31"]
    ],
    "gold":"31"
  },
  "5": {
    "name":"Cereal subtype aggregation and comparison",
    "expect_terms":["Coco Crunch","Fruit Loops"],
    "facts":[
      "Coco Crunch boxes and Fruit Loops boxes are BOTH cereal boxes.",
      "This week she bought 3 Coco Crunch boxes and 5 Fruit Loops boxes.",
      "Last week she bought 4 cereal boxes."
    ],
    "bridge":[
      "This week's total number of cereal boxes is 3 + 5.",
      "How many MORE means this week's total minus last week's total."
    ],
    "micro":[
      ["How many cereal boxes total are 3 boxes of Coco Crunch and 5 boxes of Fruit Loops?", "8"],
      ["How much more is 8 than 4?", "4"]
    ],
    "gold":"4"
  }
}
GENERIC = (
    "Answer the original question. Check units, all relevant facts, "
    "the time period each quantity refers to, and whether every required "
    "term has been included. Give only FINAL ANSWER: <number>."
)
FORMAT = "Return only FINAL ANSWER: <number>, with no explanation."

def pick_tasks(seed):
    tasks = [t for t in load_tasks(6,seed) if t["family"]=="gsm8k"]
    chosen = {}
    for t in tasks:
        key=t["id"]
        if key in CASES:
            case=CASES[key]
            if Decimal(t["gold"]) != Decimal(case["gold"]):
                raise ValueError(f"Expected gold changed for {key}: {t['gold']}")
            if not all(phrase in t["prompt"] for phrase in case["expect_terms"]):
                raise ValueError(f"Dataset or sampling mismatch for {key}")
            chosen[key]=t
    if set(chosen)!=set(CASES):
        raise ValueError("Selected problems changed; refuse to run misleading comparison")
    return chosen

def make_prompts(problem, case):
    facts="\n".join("* "+s for s in case["facts"])
    bridge="\n".join("* "+s for s in case["bridge"])
    return {
        "original": problem,
        "generic_check": problem + "\n\n" + GENERIC,
        "facts_only": problem + "\n\nVERIFIED PARTIAL FACTS (not the final answer):\n" + facts
                      + "\n\n" + FORMAT,
        "facts_plus_bridge": problem + "\n\nVERIFIED PARTIAL FACTS:\n" + facts
                             + "\n\nUSE THESE RELATIONS:\n" + bridge
                             + "\n\nCompute the remaining result. " + FORMAT,
    }

def self_test():
    assert len(CASES)==4
    for key,c in CASES.items():
        assert c["gold"] == c["micro"][-1][1],key
        assert len(make_prompts("question",c))==4
    print("PASS four annotated fact/edge cases and microtraces")

def run():
    p=argparse.ArgumentParser()
    p.add_argument("--self-test",action="store_true")
    p.add_argument("--model",default="Qwen/Qwen3-0.6B")
    p.add_argument("--seed",type=int,default=20261115)
    p.add_argument("--output",default="causal_trace_results.json")
    p.add_argument("--max-tokens",type=int,default=112)
    p.add_argument("--micro-tokens",type=int,default=48)
    args=p.parse_args()
    if args.self_test:
        self_test()
        return
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    torch.set_num_threads(2)
    chosen=pick_tasks(args.seed)
    tok=AutoTokenizer.from_pretrained(args.model)
    model=AutoModelForCausalLM.from_pretrained(args.model,dtype=torch.float32,
                                                low_cpu_mem_usage=True).eval()
    device=next(model.parameters()).device
    start=time.monotonic()
    rows=[]
    result=dict(model=args.model,config=vars(args),
                qualification="posthoc causal diagnosis on already examined cases, not blind generalization",
                rows=rows)
    def query(text,limit):
        messages=[{"role":"user","content":text}]
        ids=tok.apply_chat_template(messages,tokenize=True,
             add_generation_prompt=True,enable_thinking=False,return_tensors="pt").to(device)
        with torch.inference_mode():
            output=model.generate(ids,attention_mask=torch.ones_like(ids),do_sample=False,
                                  max_new_tokens=limit,pad_token_id=tok.eos_token_id)
        raw=tok.decode(output[0,ids.shape[1]:],skip_special_tokens=True).strip()
        return dict(raw=raw,parsed=number(raw),input_tokens=int(ids.shape[1]),
                    output_tokens=int(output.shape[1]-ids.shape[1]))
    for case_id,case in CASES.items():
        item=chosen[case_id]
        row=dict(case=case_id,name=case["name"],prompt=item["prompt"],
                 gold=item["gold"],facts=case["facts"],bridge=case["bridge"],
                 arms={},micro=[])
        for arm,prompt in make_prompts(item["prompt"],case).items():
            observed=query(prompt,args.max_tokens)
            observed["correct"]=observed["parsed"]==item["gold"]
            observed["prompt"]=prompt
            row["arms"][arm]=observed
            print(f'{case_id} {arm}: {observed["parsed"]} correct={observed["correct"]}',flush=True)
        for text,gold in case["micro"]:
            observed=query(text+"\n"+FORMAT,args.micro_tokens)
            observed["expected"]=gold
            observed["correct"]=observed["parsed"]==gold
            observed["prompt"]=text
            row["micro"].append(observed)
            print(f'  micro expected={gold} got={observed["parsed"]} correct={observed["correct"]}',flush=True)
        row["all_micro_correct"]=all(s["correct"] for s in row["micro"])
        row["first_micro_failure"]=next((i+1 for i,s in enumerate(row["micro"]) if not s["correct"]),None)
        rows.append(row)
        result["summary"]={arm:sum(int(r["arms"][arm]["correct"]) for r in rows)
                            for arm in make_prompts("",case)}
        result["summary"]["all_micro_correct"]=sum(int(r["all_micro_correct"]) for r in rows)
        result["elapsed_seconds"]=round(time.monotonic()-start,2)
        Path(args.output).write_text(json.dumps(result,indent=2))
    print(json.dumps(result["summary"],indent=2),flush=True)

if __name__=="__main__":
    run()
