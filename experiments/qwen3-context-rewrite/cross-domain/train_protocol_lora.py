#!/usr/bin/env python3
"""Real LoRA weight update of official Qwen3-0.6B for composed agent messages.

The toy test is deliberately different from GSM8K-style posthoc prompt fixes:
* 8 x 8 independent attributes, with a balanced 16-combination blind holdout.
* Two arbitrary, nonnumeric token vocabularies. All 16 primitive symbols appear
  in training; 16 *combinations* never appear in any train objective.
* ONE shared Qwen LoRA adapter learns SEND, RECEIVE, and USE as interacting
  operations. SEND output is passed without correction to USE on unseen pairs.
* Exact output parsing, held-out test checks, before/after weight-disabled
  control, optional untrained novel operations.
* The reference code is researcher-specified, NOT emergent. This is a
  learned compositional communication test, NOT proof of grokking unless a
  delayed test-accuracy jump is actually observed.

The training example's answer is never inserted in the input. Gold labels
exist only in training targets and final evaluation comparisons.
"""
import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import random
import re
import time

A = ["maple", "frost", "ocean", "cinder", "violet", "echo", "pollen", "tundra"]
B = ["marble", "orbit", "fable", "harbor", "spruce", "copper", "velvet", "comet"]
OPS = ("SEND", "RECEIVE", "USE")
SYSTEM = "You exchange compact machine-readable messages. Reply with the exact required value and nothing else."
FIXED_SEED = 20261114

def split_pairs():
    train = [(i,j) for i in range(8) for j in range(8) if (i+j)%4 != 0]
    test = [(i,j) for i in range(8) for j in range(8) if (i+j)%4 == 0]
    assert len(train)==48 and len(test)==16
    assert all(any(i==x for x,y in train) and any(j==y for x,y in train)
               for i in range(8) for j in range(8))
    assert not (set(train)&set(test))
    return train,test

def message(i,j):
    return A[i] + "|" + B[j]

def prompt(role,i=None,j=None,signal=None,op="SUM"):
    if role=="SEND":
        return f"ROLE: SEND\nOBSERVATION: X={i}; Y={j}\nMESSAGE:"
    if role=="RECEIVE":
        return f"ROLE: RECEIVE\nMESSAGE: {signal}\nOUTPUT X AND Y:"
    if role=="USE":
        return f"ROLE: USE\nMESSAGE: {signal}\nOPERATION: {op}\nANSWER:"
    raise ValueError(role)

def sample(role,i,j,op="SUM"):
    sig=message(i,j)
    inp=prompt(role,i,j,sig,op)
    target={"SEND":sig,"RECEIVE":f"X={i},Y={j}","USE":str(i+j)}[role]
    return inp,target

def match(role,pred,expected):
    clean=pred.strip().splitlines()[0].strip() if pred.strip() else ""
    # No fuzzy parsing of unfinished reasoning; strict, reproducible outputs.
    if role=="SEND":
        return clean==expected
    if role=="RECEIVE":
        return clean.replace(" ","")==expected
    if role=="USE":
        return clean==expected
    raise ValueError(role)

def self_test():
    train,test=split_pairs()
    assert len(train)==48 and len(test)==16
    for p in train+test:
        for role in OPS:
            inp,target=sample(role,*p)
            assert not target in inp if role=="SEND" else True
            assert match(role,target,target)
    assert not match("USE","2\n3","3")
    assert prompt("USE",signal="maple|orbit").endswith("ANSWER:")
    print("PASS: 48/16 disjoint compositional split, all primitives exposed, strict scoring",flush=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--self-test",action="store_true")
    ap.add_argument("--model",default="Qwen/Qwen3-0.6B")
    ap.add_argument("--steps",type=int,default=384)
    ap.add_argument("--eval-every",type=int,default=96)
    ap.add_argument("--lr",type=float,default=0.0015)
    ap.add_argument("--out",default="qwen_protocol_training")
    ap.add_argument("--seed",type=int,default=FIXED_SEED)
    ap.add_argument("--min-time-sec",type=int,default=0)
    ap.add_argument("--max-time-sec",type=int,default=2150)
    args=ap.parse_args()
    if args.self_test:
        self_test()
        return
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from peft import LoraConfig,TaskType,get_peft_model
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    train_pairs,test_pairs=split_pairs()
    outdir=Path(args.out);outdir.mkdir(parents=True,exist_ok=True)
    tok=AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token=tok.eos_token
    base=AutoModelForCausalLM.from_pretrained(args.model,
                            dtype=torch.float32,low_cpu_mem_usage=True)
    base.config.use_cache=False
    layers=range(len(base.model.layers)-4,len(base.model.layers))
    conf=LoraConfig(task_type=TaskType.CAUSAL_LM,
                    r=8,lora_alpha=16,lora_dropout=0.0,bias="none",
                    target_modules=["q_proj","v_proj","o_proj","down_proj"],
                    layers_to_transform=list(layers),layers_pattern="layers")
    net=get_peft_model(base,conf)
    trainables=[p for p in net.parameters() if p.requires_grad]
    param_count=sum(p.numel() for p in trainables)
    print(f"Trainable LoRA parameters: {param_count} / {sum(p.numel() for p in net.parameters())}",flush=True)
    optimizer=torch.optim.AdamW(trainables,lr=args.lr,weight_decay=0.0)
    all_samples=[(r,i,j) for r in OPS for i,j in train_pairs]
    rng=random.Random(args.seed+5);rng.shuffle(all_samples)
    current=0
    start=time.monotonic()
    history=[]
    log=dict(config=vars(args),protocol="Qwen3-0.6B LoRA real weight training; supervised, researcher-specified factorized code",
             vocab_X=A,vocab_Y=B,train_pairs=train_pairs,test_pairs=test_pairs,
             trainable_parameters=param_count,history=history,train_loss=[],
             status="running")
    def save_state():
        (outdir/"results.json").write_text(json.dumps(log,indent=2))
    def ids_for(messages):
        return tok.apply_chat_template(messages,tokenize=True,
            add_generation_prompt=True,enable_thinking=False,return_tensors="pt")
    def infer(text,max_tokens=20):
        x=ids_for([{"role":"system","content":SYSTEM},{"role":"user","content":text}])
        with torch.no_grad():
            y=net.generate(x,attention_mask=torch.ones_like(x),
                   do_sample=False,max_new_tokens=max_tokens,
                   pad_token_id=tok.eos_token_id)
        return tok.decode(y[0,x.shape[1]:],skip_special_tokens=True).strip()
    def evaluate(label,pairs,full=False):
        old=net.training
        net.eval()
        scores={r:{"correct":0,"total":0} for r in OPS}
        traces=[]
        with torch.inference_mode():
            for i,j in pairs:
                for role in OPS:
                    inp,expected=sample(role,i,j)
                    pred=infer(inp)
                    passed=match(role,pred,expected)
                    scores[role]["correct"]+=int(passed)
                    scores[role]["total"]+=1
                    if full or not passed:
                        traces.append(dict(role=role,i=i,j=j,prompt=inp,expected=expected,
                                           raw=pred,correct=passed))
        if old:net.train()
        return dict(label=label,scores=scores,traces=traces)
    # Baseline invokes the same model, but explicitly DISABLES adapter weights.
    with net.disable_adapter():
        before=evaluate("frozen_base_heldout",test_pairs,full=False)
    log["baseline"]=before
    print(f"FROZEN BASE HELDOUT: {before['scores']}",flush=True)
    save_state()
    def checkpoint(step):
        net.eval()
        eval_train=evaluate(f"train_{step}",train_pairs[:8])
        eval_test=evaluate(f"test_{step}",test_pairs[:8])
        row=dict(step=step,train=eval_train["scores"],heldout=eval_test["scores"],
                 elapsed_seconds=round(time.monotonic()-start,2))
        history.append(row)
        log["last_step"]=step
        log["elapsed_seconds"]=round(time.monotonic()-start,2)
        save_state()
        net.save_pretrained(outdir/"adapter")
        print("CHECKPOINT",json.dumps(row),flush=True)
    checkpoint(0)
    for step in range(1,args.steps+1):
        if step>1 and time.monotonic()-start>args.max_time_sec:
            log["stopped_due_to_time_budget"]=True
            print("STOP due to bounded runner; last adapter checkpoint retained",flush=True)
            break
        role,i,j=all_samples[current]
        current+=1
        if current==len(all_samples):
            current=0;rng.shuffle(all_samples)
        inp,target=sample(role,i,j)
        prefix=ids_for([{"role":"system","content":SYSTEM},{"role":"user","content":inp}])[0]
        suffix=tok.encode(target,add_special_tokens=False)+[tok.eos_token_id]
        # Teacher-force target only; never supervise on prompt tokens.
        allids=torch.cat((prefix,torch.tensor(suffix,dtype=torch.long)))
        labels=torch.cat((torch.full_like(prefix,-100),torch.tensor(suffix,dtype=torch.long)))
        net.train();optimizer.zero_grad(set_to_none=True)
        output=net(input_ids=allids.unsqueeze(0),attention_mask=torch.ones((1,len(allids)),dtype=torch.long),
                   labels=labels.unsqueeze(0),use_cache=False)
        loss=output.loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainables,1.0)
        optimizer.step()
        if step==1 or step%24==0:
            point=dict(step=step,loss=round(float(loss.detach()),5),
                       seconds=round(time.monotonic()-start,1))
            log["train_loss"].append(point)
            print("TRAIN",json.dumps(point),flush=True)
        if step%args.eval_every==0:
            checkpoint(step)
    # Save at the actual completed step even when preempted.
    if log.get("last_step")!=step:
        checkpoint(step)
    final=evaluate("trained_adapter_heldout",test_pairs,full=True)
    with net.disable_adapter():
        clean=evaluate("frozen_base_heldout_exact_same_test",test_pairs,full=True)
    net.eval()
    chained=[]
    with torch.inference_mode():
        for i,j in test_pairs:
            sender_raw=infer(prompt("SEND",i=i,j=j))
            use_raw=infer(prompt("USE",signal=sender_raw))
            recv_raw=infer(prompt("RECEIVE",signal=sender_raw))
            chained.append(dict(i=i,j=j,
                    sender_raw=sender_raw,receiver_raw=recv_raw,use_raw=use_raw,
                    send_correct=match("SEND",sender_raw,message(i,j)),
                    receive_correct=match("RECEIVE",recv_raw,f"X={i},Y={j}"),
                    downstream_correct=match("USE",use_raw,str(i+j))))
    log["final"]=dict(test=final,base=clean,
                       end_to_end=dict(total=len(chained),
                         correct=sum(int(r["send_correct"] and r["receive_correct"] and r["downstream_correct"]) for r in chained),
                         rows=chained))
    net.save_pretrained(outdir/"adapter")
    log["status"]="completed";log["elapsed_seconds"]=round(time.monotonic()-start,1)
    save_state()
    print("FINAL",json.dumps(dict(heldout=final["scores"],baseline=clean["scores"],
                                 chained_correct=log["final"]["end_to_end"]["correct"],
                                 seconds=log["elapsed_seconds"])),flush=True)

if __name__=="__main__":
    main()
