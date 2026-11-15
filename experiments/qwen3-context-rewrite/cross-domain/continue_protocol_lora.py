#!/usr/bin/env python3
"""Continued REAL LoRA weight training for communication composition in Qwen3-0.6B.

Starts with the previous trained adapter rather than resetting its weights.
The earlier run learned RECEIVE partially (8/16 unseen), but SEND and USE
collapsed to constants. Corrects gradient conflict by using a BALANCED
minibatch that jointly updates the same adapter on SEND, RECEIVE and numeric
CALCULATE, while rehearsing independent symbol encoding/decoding constituents.

This is one shared Qwen adapter. At evaluation, actual sender output goes to
receiver; the receiver output goes to a numeric operation. No oracle value
is inserted, and the same original 16 held-out *combinations* remain withheld.

Claims must distinguish supervised factorized-code learning from emergent
protocol discovery; a grokking transition requires an actual delayed jump.
"""
import argparse
from contextlib import nullcontext
import hashlib
import json
import math
import random
import re
import time
from pathlib import Path

from train_protocol_lora import A,B,split_pairs,message,SYSTEM,match

ROLES=("SEND","RECEIVE","CALC","ENC_X","ENC_Y","DEC_X","DEC_Y")
REGEX_DECODE=re.compile(r"^X=([0-7]),Y=([0-7])$")

def task(role,i,j):
    if role=="SEND":
        return f"ROLE: SEND\nOBSERVATION: X={i}; Y={j}\nMESSAGE:",message(i,j)
    if role=="RECEIVE":
        return f"ROLE: RECEIVE\nMESSAGE: {message(i,j)}\nOUTPUT X AND Y:",f"X={i},Y={j}"
    if role=="CALC":
        return f"ROLE: CALCULATE\nVALUES: X={i},Y={j}\nOPERATION: SUM\nANSWER:",str(i+j)
    if role=="ENC_X":
        return f"ROLE: ENCODE X\nVALUE: X={i}\nSYMBOL:",A[i]
    if role=="ENC_Y":
        return f"ROLE: ENCODE Y\nVALUE: Y={j}\nSYMBOL:",B[j]
    if role=="DEC_X":
        return f"ROLE: DECODE X\nSYMBOL: {A[i]}\nVALUE:",str(i)
    if role=="DEC_Y":
        return f"ROLE: DECODE Y\nSYMBOL: {B[j]}\nVALUE:",str(j)
    raise ValueError(role)

def correct(role,raw,expected):
    clean=raw.strip().splitlines()[0].strip() if raw.strip() else ""
    if role=="RECEIVE":
        return clean.replace(" ","")==expected
    return clean==expected

def simple_test():
    train,heldout=split_pairs()
    assert len(train)==48 and len(heldout)==16
    for i,j in train+heldout:
        for r in ROLES:
            p,t=task(r,i,j)
            assert p and t and correct(r,t,t)
    assert REGEX_DECODE.fullmatch("X=2,Y=5")
    assert not REGEX_DECODE.fullmatch("X=5,Y=20")
    print("PASS: balanced three-role objective + four constituent roles, same 48/16 split",flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model",default="Qwen/Qwen3-0.6B")
    p.add_argument("--initial-adapter",default="previous/adapter")
    p.add_argument("--out",default="qwen_protocol_continued")
    p.add_argument("--steps",type=int,default=900)
    p.add_argument("--eval-every",type=int,default=300)
    p.add_argument("--lr",type=float,default=0.00045)
    p.add_argument("--max-seconds",type=int,default=3200)
    p.add_argument("--seed",type=int,default=20261114)
    p.add_argument("--self-test",action="store_true")
    args=p.parse_args()
    if args.self_test:
        simple_test()
        return
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from peft import PeftModel
    torch.set_num_threads(2)
    torch.manual_seed(args.seed+1)
    rng=random.Random(args.seed+97)
    train_pairs,heldout=split_pairs()
    adapter_path=Path(args.initial_adapter)
    weight_file=adapter_path/"adapter_model.safetensors"
    if not weight_file.is_file():
        raise FileNotFoundError(f"Cannot continue without the trained weights: {weight_file}")
    source_hash=hashlib.sha256(weight_file.read_bytes()).hexdigest()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    tok=AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    base=AutoModelForCausalLM.from_pretrained(args.model,dtype=torch.float32,low_cpu_mem_usage=True)
    base.config.use_cache=False
    net=PeftModel.from_pretrained(base,str(adapter_path),is_trainable=True)
    trainables=[(name,t) for name,t in net.named_parameters() if t.requires_grad]
    assert trainables, "Loaded adapter but no trainable weights"
    init={name:t.detach().clone() for name,t in trainables}
    opt=torch.optim.AdamW([t for name,t in trainables],lr=args.lr,weight_decay=.001)
    init_clock=time.monotonic()
    result=dict(protocol="continued Qwen3-0.6B trained PEFT adapter; balanced primitive/composition multitask",
                config=vars(args),source_adapter_sha256=source_hash,
                train_pairs=train_pairs,heldout_pairs=heldout,
                trainable_parameters=sum(t.numel() for _,t in trainables),
                checkpoints=[],losses=[],status="training")
    def save():
        (out/"results.json").write_text(json.dumps(result,indent=2))
    def tokens(prompt):
        return tok.apply_chat_template(
          [{"role":"system","content":SYSTEM},{"role":"user","content":prompt}],
          tokenize=True,add_generation_prompt=True,enable_thinking=False,return_tensors="pt")[0]
    def gen(prompt,max_new_tokens=24):
        pre=tokens(prompt)
        with torch.inference_mode():
            o=net.generate(pre.unsqueeze(0),
                    attention_mask=torch.ones((1,len(pre)),dtype=torch.long),
                    do_sample=False,max_new_tokens=max_new_tokens,
                    pad_token_id=tok.eos_token_id)
        return tok.decode(o[0,len(pre):],skip_special_tokens=True).strip()
    def evaluate(pairs,label,chain=False):
        net.eval()
        totals={k:dict(correct=0,total=0) for k in ("SEND","RECEIVE","CALC")}
        traced=[]
        with torch.inference_mode():
            for i,j in pairs:
                rec={"i":i,"j":j}
                for role in totals:
                    inp,expected=task(role,i,j)
                    got=gen(inp)
                    passed=correct(role,got,expected)
                    totals[role]["total"]+=1
                    totals[role]["correct"]+=int(passed)
                    rec[role]=dict(expected=expected,raw=got,correct=passed)
                if chain:
                    send=rec["SEND"]["raw"]
                    decode=gen(f"ROLE: RECEIVE\nMESSAGE: {send}\nOUTPUT X AND Y:")
                    pair=REGEX_DECODE.fullmatch(decode.replace(" ","").strip())
                    calc=(gen(f"ROLE: CALCULATE\nVALUES: X={pair.group(1)},Y={pair.group(2)}\nOPERATION: SUM\nANSWER:")
                          if pair else None)
                    rec["end_to_end"]=dict(sent=send,decoded=decode,calculated=calc,
                                            correct=calc==str(i+j) and decode.replace(" ","").strip()==f"X={i},Y={j}")
                traced.append(rec)
        return dict(label=label,scores=totals,
                    end_to_end_correct=sum(int(x.get("end_to_end",{}).get("correct",False)) for x in traced) if chain else None,
                    traces=traced)
    def checkpoint(step):
        net.eval()
        tr=evaluate(train_pairs[:12],f"train_at_{step}")
        te=evaluate(heldout[:12],f"heldout_at_{step}")
        obj=dict(step=step,train=tr["scores"],test=te["scores"],
                 elapsed_seconds=round(time.monotonic()-init_clock,2))
        result["checkpoints"].append(obj)
        result["last_step"]=step
        save()
        net.save_pretrained(out/"adapter")
        print("CHECKPOINT",json.dumps(obj),flush=True)
    def batch(inputs):
        seqs=[]
        for role,i,j in inputs:
            inp,target=task(role,i,j)
            pre=tokens(inp)
            suf=tok.encode(target,add_special_tokens=False)+[tok.eos_token_id]
            ids=torch.cat((pre,torch.tensor(suf,dtype=torch.long)))
            labels=torch.cat((torch.full_like(pre,-100),torch.tensor(suf,dtype=torch.long)))
            seqs.append((ids,labels))
        maxlen=max(len(ids) for ids,_ in seqs)
        ids=torch.full((len(seqs),maxlen),tok.pad_token_id,dtype=torch.long)
        labs=torch.full((len(seqs),maxlen),-100,dtype=torch.long)
        mask=torch.zeros((len(seqs),maxlen),dtype=torch.long)
        for row,(x,y) in enumerate(seqs):
            length=len(x);ids[row,:length]=x;labs[row,:length]=y;mask[row,:length]=1
        return dict(input_ids=ids,attention_mask=mask,labels=labs,use_cache=False)
    print("RESUMED",json.dumps(dict(source_sha256=source_hash[:16],
                                 parameters=result["trainable_parameters"],
                                 train=len(train_pairs),test=len(heldout))),flush=True)
    checkpoint(0)
    last=0
    for step in range(1,args.steps+1):
        if step>1 and time.monotonic()-init_clock>args.max_seconds:
            result["stopped_due_to_time_budget"]=True
            break
        # Balance SEND, RECEIVE, CALC on EACH update. Rehearse one constituent
        # encode/decode operation to prevent independent symbol maps from dying.
        kind=ROLES[3+(step-1)%4]
        pairs=[rng.choice(train_pairs) for _ in range(3)]
        sample=[("SEND",*pairs[0]),("RECEIVE",*pairs[1]),("CALC",*pairs[2]),
                (kind,*rng.choice(train_pairs))]
        net.train()
        opt.zero_grad(set_to_none=True)
        loss=net(**batch(sample)).loss
        if not torch.isfinite(loss):raise RuntimeError(f"Invalid loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_([t for _,t in trainables],1.0)
        opt.step()
        if step==1 or step%40==0:
            point=dict(step=step,loss=round(float(loss.detach()),5),
                       seconds=round(time.monotonic()-init_clock,1))
            result["losses"].append(point)
            print("TRAIN",json.dumps(point),flush=True)
        last=step
        if step%args.eval_every==0:
            checkpoint(step)
    if result["last_step"]!=last:
        checkpoint(last)
    after=evaluate(heldout,"trained_heldout_full",chain=True)
    before=evaluate(train_pairs,"trained_seen_full",chain=False)
    diff=sum(float(((p-init[name])**2).sum()) for name,p in trainables)
    result["weight_update_l2"]=math.sqrt(diff)
    result["final"]=dict(heldout=after,seen=before)
    result["status"]="completed"
    result["elapsed_seconds"]=round(time.monotonic()-init_clock,2)
    net.save_pretrained(out/"adapter")
    save()
    print("FINAL",json.dumps(dict(step=last,heldout=after["scores"],
                                  seen=before["scores"],
                                  end_to_end=after["end_to_end_correct"],
                                  weight_update_l2=result["weight_update_l2"],
                                  seconds=result["elapsed_seconds"])),flush=True)

if __name__=="__main__":
    main()
