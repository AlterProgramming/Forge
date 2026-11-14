#!/usr/bin/env python3
"""Inference-only Qwen3 text working-memory rewrite probe.

This is NOT latent-state editing or training. Both conditions get the
same Countdown-4 problems, step budgets, and working-memory capacity.
"""
import argparse, ast, json, random, re, time
from collections import Counter
from fractions import Fraction
from pathlib import Path

BINOPS = {ast.Add:lambda a,b:a+b, ast.Sub:lambda a,b:a-b,
          ast.Mult:lambda a,b:a*b, ast.Div:lambda a,b:a/b}

def verify(expression, nums, target):
    if not expression or len(expression)>300:
        return False, "empty or excessive expression"
    used=[]
    def visit(n):
        if isinstance(n,ast.Expression): return visit(n.body)
        if isinstance(n,ast.Constant) and type(n.value)==int:
            used.append(n.value)
            return Fraction(n.value)
        if isinstance(n,ast.BinOp) and type(n.op) in BINOPS:
            return BINOPS[type(n.op)](visit(n.left),visit(n.right))
        raise ValueError("invalid syntax")
    try:
        value=visit(ast.parse(expression,mode="eval"))
        if Counter(used)!=Counter(nums):return False, "incorrect operands"
        return value==target, str(value)
    except (SyntaxError,ValueError,ZeroDivisionError,OverflowError,RecursionError) as e:
        return False, str(e)

def decode_json(s):
    d=json.JSONDecoder()
    candidates=[]
    for i,c in enumerate(s):
        if c=="{":
            try:
                obj,_=d.raw_decode(s[i:])
                if isinstance(obj,dict):candidates.append(obj)
            except json.JSONDecodeError:pass
    return candidates[-1] if candidates else {}

def memory_trim(tok,text,budget,keep_tail=True):
    ids=tok.encode(text,add_special_tokens=False)
    if len(ids)>budget:
        ids=ids[-budget:] if keep_tail else ids[:budget]
    return tok.decode(ids,skip_special_tokens=True)

def prompt(ex,mode,state,step,steps,budget):
    directive=(
        "REWRITE: produce a new useful working representation replacing ALL prior working memory. "
        "You can reorganize facts, introduce invariants, change algebraic notation or discard dead ends. "
        "The new state is the ONLY state kept for the next round."
        if mode=="rewrite" else
        "APPEND: produce exactly one NEW thought to append to the existing working memory. "
        "You cannot revise previous thoughts; old thoughts fall off when the budget is reached."
    )
    return (
        "Solve Countdown-4 using each number exactly once with + - * / and parentheses. "
        "No other operations or concatenation.\n"
        f"Numbers: {ex['numbers']} | Target: {ex['target']}\n"
        f"Round {step}/{steps}, memory limit {budget} tokens.\n"
        f"{directive}\nWorking memory:\n{state or '(empty)'}\n"
        'Give one JSON object only: {"state":"your working representation or thought",'
        '"answer":"exact arithmetic expression if solved, else empty string"}.'
    )

def run_one(ex,mode,model,tok,device,args):
    import torch
    state=""
    trace=[]
    in_tokens=out_tokens=0
    for step in range(1,args.steps+1):
        messages=[{"role":"user","content":prompt(ex,mode,state,step,args.steps,args.memory_tokens)}]
        ids=tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,
                                    enable_thinking=False,return_tensors="pt").to(device)
        with torch.inference_mode():
            y=model.generate(ids,do_sample=False,max_new_tokens=args.max_new_tokens,
                             pad_token_id=tok.eos_token_id)
        raw=tok.decode(y[0,ids.shape[1]:],skip_special_tokens=True)
        parsed=decode_json(raw)
        answer=str(parsed.get("answer") or "").strip()
        new_state=str(parsed.get("state") or "").strip()
        valid,why=verify(answer,ex["numbers"],ex["target"])
        if mode=="rewrite":
            state=memory_trim(tok,new_state,args.memory_tokens,False)
        else:
            state=memory_trim(tok,(state+"\n"+new_state).strip(),args.memory_tokens,True)
        a={"round":step,"raw":raw,"answer":answer,"parsed_state":new_state,
           "valid":valid,"validator":why,"memory_after":state,
           "input_tokens":int(ids.shape[1]),"output_tokens":int(y.shape[1]-ids.shape[1])}
        trace.append(a)
        in_tokens+=a["input_tokens"]
        out_tokens+=a["output_tokens"]
        print(f"  {mode} round={step} valid={valid} input={a['input_tokens']} generated={a['output_tokens']}",flush=True)
        if valid:break
    return {"solved":any(t["valid"] for t in trace),"input_tokens":in_tokens,
            "generated_tokens":out_tokens,"trace":trace}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model",default="Qwen/Qwen3-0.6B")
    p.add_argument("--examples",type=int,default=2)
    p.add_argument("--steps",type=int,default=2)
    p.add_argument("--max-new-tokens",type=int,default=96)
    p.add_argument("--memory-tokens",type=int,default=64)
    p.add_argument("--output",default="qwen3_results.json")
    p.add_argument("--self-test",action="store_true")
    args=p.parse_args()
    if args.self_test:
        for expr,nums,target,expected in [
            ("(8-2)*(3+1)",[8,2,3,1],24,True),
            ("8*3",[8,2,3,1],24,False),
            ("(7+1)*(6/2)",[7,1,6,2],24,True)]:
            actual,_=verify(expr,nums,target)
            assert actual==expected,(expr,actual)
        print("validator self-test passed",flush=True)
        return
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    torch.set_num_threads(2)
    print("Downloading/loading",args.model,flush=True)
    tok=AutoTokenizer.from_pretrained(args.model)
    model=AutoModelForCausalLM.from_pretrained(args.model,torch_dtype=torch.float32,
                                              low_cpu_mem_usage=True).eval()
    device=next(model.parameters()).device
    examples=[
        {"numbers":[8,2,3,1],"target":24},
        {"numbers":[7,1,6,2],"target":24},
        {"numbers":[5,9,3,2],"target":24}
    ][:args.examples]
    output={"config":vars(args),"model":args.model,"results":[]}
    started=time.monotonic()
    for i,ex in enumerate(examples):
        print("Problem",i+1,ex,flush=True)
        row={"problem":ex}
        for mode in ("append","rewrite"):
            row[mode]=run_one(ex,mode,model,tok,device,args)
        output["results"].append(row)
        Path(args.output).write_text(json.dumps(output,indent=2))
    output["elapsed_sec"]=round(time.monotonic()-started,2)
    output["summary"]={mode:{"solved":sum(r[mode]["solved"] for r in output["results"]),
                            "total":len(output["results"]),
                            "tokens":sum(r[mode]["input_tokens"]+r[mode]["generated_tokens"] for r in output["results"])}
                       for mode in ("append","rewrite")}
    Path(args.output).write_text(json.dumps(output,indent=2))
    print("RESULT",json.dumps(output["summary"]),flush=True)
    print("Elapsed seconds",output["elapsed_sec"],flush=True)

if __name__=="__main__":
    main()
