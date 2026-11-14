#!/usr/bin/env python3
"""Test the smallest Qwen-to-Qwen communication boundary directly.

Sender = frozen Qwen3-0.6B generates one intermediate fact for a public question.
Receiver = *the same frozen Qwen weights*, independently called on the question
plus a message slot. The only experimental variable for receiver prompts is
the message slot's CONTENT: empty, actual signal, 1/4/8 tokenizer tokens,
or a sender signal from a different question (shuffled control).

Measure whether messages change decisions, whether the changes help, how
many communication tokens are necessary, and full inference costs.
NO program decomposition is supplied; NO oracle answers enter prompts.
This is an experiment about an observable textual communication channel,
NOT latent internal activations, training, or guaranteed transfer.
"""
import argparse
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
import random
import time

from boost import number, choice

ARMS = ("none", "one", "four", "eight", "full", "shuffled")
SYSTEM_SENDER = ("Give exactly ONE useful intermediate calculation, constraint "
                 "or relationship for solving the problem. Do not give the final "
                 "answer. Do not add analysis or explanations. Keep it short.")
SYSTEM_RECEIVER = "Solve the question. Check whether a supplied intermediate claim is relevant and correct."

def public_tasks(family, n, seed):
    from datasets import load_dataset
    rng = random.Random(seed + (0 if family == "gsm8k" else 2718))
    if family == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test")
        pool = [ds[i] for i in rng.sample(range(len(ds)), n)]
        tasks = []
        for x in pool:
            gold = number(x["answer"].split("####")[-1])
            if gold is None:
                raise ValueError("Unparseable GSM8K answer")
            tasks.append(dict(family=family, id=x["question"][:48],
                              question=x["question"], gold=gold))
        return tasks
    if family == "arc":
        ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
        pool = [ds[i] for i in rng.sample(range(len(ds)), n)]
        tasks=[]
        for x in pool:
            labs=[str(z) for z in x["choices"]["label"]]
            opts="\n".join(l + ". " + t for l,t in zip(labs,x["choices"]["text"]))
            tasks.append(dict(family=family,id=x["id"],
                              question=x["question"] + "\n" + opts,
                              labels=labs,gold=str(x["answerKey"])))
        return tasks
    raise ValueError(family)

def normalize(raw):
    return raw.replace("<|im_end|>","").strip()

def sender_user(question):
    return "QUESTION:\n" + question + "\n\nWrite the single most useful intermediate fact. No final answer."

def receiver_user(question, channel, family):
    fmt = ("FINAL ANSWER: <number>" if family == "gsm8k"
           else "FINAL ANSWER: <option label>")
    return ("QUESTION:\n"+question+"\n\n"
            +"PRIOR COMPUTATION MESSAGE (may be absent, incomplete or wrong):\n"
            +channel+"\n\n"
            +"Use it only if justified by the question. Respond with "+fmt+".")

def signal_variants(signals, tokenizer):
    """Content transformation only. Same receiver instructions for all arms."""
    if not signals:
        return []
    outputs=[]
    for i,sig in enumerate(signals):
        ids=tokenizer.encode(sig,add_special_tokens=False)
        variants={"none":"(none)",
                  "one":tokenizer.decode(ids[:1],skip_special_tokens=True).strip(),
                  "four":tokenizer.decode(ids[:4],skip_special_tokens=True).strip(),
                  "eight":tokenizer.decode(ids[:8],skip_special_tokens=True).strip(),
                  "full":sig,
                  "shuffled":signals[(i+1)%len(signals)]}
        outputs.append(variants)
    return outputs

def numeric_equal(a,b):
    if a is None:
        return False
    try:
        return Decimal(str(a)) == Decimal(str(b))
    except (InvalidOperation, ValueError):
        return False

def score(family, answer, task):
    if family == "gsm8k":
        return numeric_equal(answer,task["gold"])
    return answer == task["gold"]

def decode_answer(family, raw, task):
    return number(raw) if family == "gsm8k" else choice(raw,task["labels"])

def rollup(rows, family):
    sums={}
    for arm in ARMS:
        rs=[x["receiver"][arm] for x in rows]
        sgen=sum(x["generated_tokens"] for x in rows)
        sin=sum(x["input_tokens"] for x in rows)
        sums[arm]=dict(n=len(rs),correct=sum(int(x["correct"]) for x in rs),
                        answer_changes_from_none=sum(int(x["answer"]!=r["receiver"]["none"]["answer"])
                                                     for x,r in zip(rs,rows)),
                        sender_generated_tokens=0 if arm=="none" else sgen,
                        receiver_generated_tokens=sum(x["generated_tokens"] for x in rs),
                        sender_input_tokens=0 if arm=="none" else sin,
                        receiver_input_tokens=sum(x["input_tokens"] for x in rs),
                        message_tokens=sum(x["message_tokens"] for x in rs))
        sums[arm]["total_generated_tokens"]=sums[arm]["sender_generated_tokens"] + sums[arm]["receiver_generated_tokens"]
    return sums

def self_test():
    class T:
        def encode(self,s,add_special_tokens=False):
            return s.split()
        def decode(self,ids,skip_special_tokens=True):
            return " ".join(ids)
    s=["A B C D E F G H I", "X Y Z"]
    v=signal_variants(s,T())
    assert v[0]["one"]=="A" and v[0]["four"]=="A B C D"
    assert v[0]["shuffled"]=="X Y Z"
    assert numeric_equal("3E+1","30")
    assert not numeric_equal("3E+1","3")
    assert set(v[0])==set(ARMS)
    assert "(none)" in receiver_user("Question","(none)","gsm8k")
    print("PASS: signal token cuts, per-problem negative control, semantic numeric scoring",flush=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",default="Qwen/Qwen3-0.6B")
    ap.add_argument("--family",choices=["gsm8k","arc"],default="gsm8k")
    ap.add_argument("--n",type=int,default=6)
    ap.add_argument("--seed",type=int,default=20261116)
    ap.add_argument("--sender-tokens",type=int,default=40)
    ap.add_argument("--receiver-tokens",type=int,default=112)
    ap.add_argument("--output",default="message_causality.json")
    ap.add_argument("--self-test",action="store_true")
    args=ap.parse_args()
    if args.self_test:
        self_test()
        return
    if args.n < 2:
        ap.error("n >= 2 required for shuffled-message negative control")
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    torch.set_num_threads(2)
    tasks=public_tasks(args.family,args.n,args.seed)
    tok=AutoTokenizer.from_pretrained(args.model)
    model=AutoModelForCausalLM.from_pretrained(args.model,dtype=torch.float32,
                                               low_cpu_mem_usage=True).eval()
    device=next(model.parameters()).device
    def invoke(role,text,max_new_tokens):
        msgs=[{"role":"system","content":SYSTEM_SENDER if role=="sender" else SYSTEM_RECEIVER},
              {"role":"user","content":text}]
        ids=tok.apply_chat_template(msgs,tokenize=True,add_generation_prompt=True,
                 enable_thinking=False,return_tensors="pt").to(device)
        started=time.monotonic()
        with torch.inference_mode():
            out=model.generate(ids,attention_mask=torch.ones_like(ids),
                               do_sample=False,max_new_tokens=max_new_tokens,
                               pad_token_id=tok.eos_token_id)
        raw=tok.decode(out[0,ids.shape[1]:],skip_special_tokens=True)
        return dict(raw=raw,content=normalize(raw),input_tokens=int(ids.shape[1]),
                    generated_tokens=int(out.shape[1]-ids.shape[1]),
                    elapsed_seconds=round(time.monotonic()-started,4))
    # Generate all sender messages first, so each receiver experiment can be
    # compared against a true cross-question shuffled-signal control.
    messages=[invoke("sender",sender_user(t["question"]),args.sender_tokens) for t in tasks]
    signals=[x["content"].strip().split("\n")[0] for x in messages]
    variants=signal_variants(signals,tok)
    rows=[]
    started=time.monotonic()
    for i,task in enumerate(tasks):
        row=dict(id=task["id"],family=args.family,question=task["question"],
                 gold_eval_only=task["gold"],sender=messages[i],
                 signal=signals[i],signal_tokens=len(tok.encode(signals[i],add_special_tokens=False)),
                 receiver={})
        for arm in ARMS:
            msg=variants[i][arm]
            ret=invoke("receiver",receiver_user(task["question"],msg,args.family),
                       args.receiver_tokens)
            ret["message"]=msg
            ret["message_tokens"]=len(tok.encode(msg,add_special_tokens=False))
            ret["answer"]=decode_answer(args.family,ret["content"],task)
            ret["correct"]=score(args.family,ret["answer"],task)
            row["receiver"][arm]=ret
            print(f'{args.family} {i+1}/{len(tasks)} arm={arm} '
                  f'signal={ret["message_tokens"]} answer={ret["answer"]} '
                  f'correct={ret["correct"]}',flush=True)
        rows.append(row)
        result=dict(model=args.model,config=vars(args),protocol=(
                        "same frozen Qwen weights; same receiver question+instructions; "
                        "only message content varies; no oracle intermediate answers"),
                    rows=rows,summary=rollup(rows,args.family),
                    elapsed_seconds=round(time.monotonic()-started,2))
        Path(args.output).write_text(json.dumps(result,indent=2))
    print(json.dumps(result["summary"],indent=2),flush=True)

if __name__=="__main__":
    main()
