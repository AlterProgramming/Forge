#!/usr/bin/env python3
"""Measure communication as information transfer, separate from reasoning.

For independently sampled official GSM8K questions, hide the first numeric
quantity from the receiver. Sender Qwen sees the full question and must emit
ONLY that missing value. Receiver Qwen sees the masked question plus a signal.

Stage 1: can receiver recover the missing value from the signal? This isolates
the elementary communication channel.
Stage 2: can the same receiver use the restored value to solve the original
word problem? This isolates the DIFFERENT ability of composing information.

Negative controls: no signal, another question's signal, sender signal.
The model's own sender value is NEVER corrected with the true value.
This is an instrumented necessity test, not a production accuracy claim.
"""
import argparse
from decimal import Decimal
import json
from pathlib import Path
import random
import re
import time

from boost import number
from message_causality import public_tasks, numeric_equal

QUANTITY = re.compile(r"(?<![A-Za-z0-9])\d+(?:,\d{3})*(?:\.\d+)?")

def mask_first(question):
    match=QUANTITY.search(question)
    if match is None:
        return None
    return question[:match.start()] + "[MISSING]" + question[match.end():], match.group(0)

def self_test():
    x=mask_first("15 minutes for each of 80 rooms.")
    assert x==("[MISSING] minutes for each of 80 rooms.","15"),x
    assert numeric_equal("1E+1","10")
    assert not numeric_equal("1E+1","1")
    print("PASS: same-position masked input, preserved numeric meaning",flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model",default="Qwen/Qwen3-0.6B")
    p.add_argument("--n",type=int,default=5)
    p.add_argument("--seed",type=int,default=20261117)
    p.add_argument("--output",default="channel_necessity_results.json")
    p.add_argument("--self-test",action="store_true")
    args=p.parse_args()
    if args.self_test:
        self_test()
        return
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    torch.set_num_threads(2)
    tasks=public_tasks("gsm8k",args.n,args.seed)
    masked=[mask_first(t["question"]) for t in tasks]
    if any(x is None for x in masked):
        raise RuntimeError("Cannot mask one question; resample before inference, not after results")
    tok=AutoTokenizer.from_pretrained(args.model)
    model=AutoModelForCausalLM.from_pretrained(args.model,dtype=torch.float32,
                                               low_cpu_mem_usage=True).eval()
    device=next(model.parameters()).device
    def ask(text,limit):
        ids=tok.apply_chat_template([{"role":"user","content":text}],
             tokenize=True,add_generation_prompt=True,enable_thinking=False,
             return_tensors="pt").to(device)
        begin=time.monotonic()
        with torch.inference_mode():
            result=model.generate(ids,attention_mask=torch.ones_like(ids),
                                  do_sample=False,max_new_tokens=limit,
                                  pad_token_id=tok.eos_token_id)
        raw=tok.decode(result[0,ids.shape[1]:],skip_special_tokens=True).strip()
        return dict(raw=raw,parsed=number(raw),
                    input_tokens=int(ids.shape[1]),
                    generated_tokens=int(result.shape[1]-ids.shape[1]),
                    elapsed_seconds=round(time.monotonic()-begin,3))
    senders=[]
    signals=[]
    for task,(incomplete,hidden) in zip(tasks,masked):
        prompt=("A word problem is shown in two forms. "
                "The second has exactly one hidden number. "
                "Transmit JUST that hidden number with no extra words.\n"
                "ORIGINAL:\n"+task["question"]+"\n\n"
                "RECEIVER VERSION:\n"+incomplete+"\n"
                "Reply with one numerical value.")
        sender=ask(prompt,48)
        sender["actual_hidden_eval_only"]=hidden
        sender["correct"]=numeric_equal(sender["parsed"],hidden)
        sender["prompt"]=prompt
        senders.append(sender)
        signals.append(format(Decimal(sender["parsed"]), "f") if sender["parsed"] is not None else None)
    started=time.monotonic()
    rows=[]
    for i,(task,(incomplete,hidden)) in enumerate(zip(tasks,masked)):
        conditions={"absent":"(none)",
                    "own":signals[i] or "(unavailable)",
                    "shuffled":signals[(i+1)%len(signals)] or "(unavailable)",
                    "direct_sender":signals[i] or "(unavailable)",
                    "direct_source":hidden}
        row=dict(id=task["id"],question_full_eval_only=task["question"],
                 question_masked=incomplete,hidden_eval_only=hidden,
                 gold_eval_only=task["gold"],sender=senders[i],conditions={})
        for label,msg in conditions.items():
            if label in ("direct_sender","direct_source"):
                # The communication primitive is just a lossless state write.
                # No second Qwen call is needed to decide where the value goes:
                # the position is already known in the receiver's state.
                patched=incomplete.replace("[MISSING]",msg)
                echo_prompt=("RECEIVED_VALUE="+msg+"\n"
                             "Read only RECEIVED_VALUE. Copy its exact numeric value. "
                             "Output the number only, no calculation.")
                copy=ask(echo_prompt,16)
                solve=ask(patched+"\nGive FINAL ANSWER: followed by the numerical answer.",112)
            else:
                prefix=("PROBLEM WITH ONE VALUE MISSING:\n"+incomplete+
                        "\n\nMESSAGE FROM ANOTHER COMPUTATION:\n"+msg+"\n\n")
                copy=ask(prefix+"What exact number belongs in [MISSING]? "
                         "Reply with one numerical value.",48)
                solve=ask(prefix+"Solve the problem by using the message only if it "
                         "supplies a legitimate missing value. "
                         "Output only FINAL ANSWER: followed by the numerical answer.",112)
            copy["correct"]=numeric_equal(copy["parsed"],hidden)
            copy["message_exact"]=numeric_equal(msg,hidden)
            solve["correct"]=numeric_equal(solve["parsed"],task["gold"])
            row["conditions"][label]=dict(message=msg,copy=copy,solve=solve)
            print(f'{i+1}/{len(tasks)} {label} sender_correct={row["sender"]["correct"]} '
                  f'copy={copy["correct"]} solve={solve["correct"]}',flush=True)
        rows.append(row)
        summary={}
        for label in conditions:
            summary[label]=dict(n=len(rows),
               transmitted_correct=sum(int(r["sender"]["correct"]) for r in rows) if label in ("own","direct_sender") else None,
               value_recovered=sum(int(r["conditions"][label]["copy"]["correct"]) for r in rows),
               word_problem_solved=sum(int(r["conditions"][label]["solve"]["correct"]) for r in rows),
               output_tokens=sum(r["conditions"][label][k]["generated_tokens"] for r in rows for k in ("copy","solve"))
                             + (sum(r["sender"]["generated_tokens"] for r in rows) if label=="own" else 0))
        Path(args.output).write_text(json.dumps(dict(model=args.model,
               protocol="sender has missing input; no oracle corrections; same receiver prompt except signal",
               config=vars(args),rows=rows,summary=summary,
               elapsed_seconds=round(time.monotonic()-started,2)),indent=2))
    print(json.dumps(summary,indent=2),flush=True)

if __name__=="__main__":
    main()
