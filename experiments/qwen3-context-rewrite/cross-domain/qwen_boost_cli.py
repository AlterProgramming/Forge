#!/usr/bin/env python3
"""Reusable Qwen3-0.6B two-phase inference for arbitrary user prompts.

No extra weights and no problem-family-specific graph. A native thinking pass
is allowed to work for a bounded number of tokens; if the thinking pass never
reaches a final response, a non-thinking pass is required to finish using
the partial reasoning. Verified experimentally on GSM8K, ARC and MBPP;
generalization to arbitrary new prompts is not yet established.
"""
import argparse
import json
import sys
import time
from pathlib import Path

def strip_end(text):
    return text.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()

def interpret_thinking(text):
    """Return (answer, unfinished, scratchpad)."""
    if "</think>" in text:
        before, after = text.split("</think>", 1)
        final = strip_end(after)
        if final:
            return final, False, before.replace("<think>", "").strip()
        return "", True, before.replace("<think>", "").strip()
    if "<think>" in text:
        return "", True, text.split("<think>",1)[-1].strip()
    # Some Qwen turns directly output a final response in thinking mode.
    return strip_end(text), False, ""

def completion_prompt(user_text, scratchpad, answer_format):
    format_instructions = {
        "text": "Respond directly and completely to the user's request.",
        "number": "Output exactly FINAL ANSWER: <number>.",
        "choice": "Output exactly FINAL ANSWER: <letter>.",
        "python": "Output only complete runnable Python code, not Markdown.",
    }
    return (
        "A previous reasoning pass ran out of tokens. "
        "Use any relevant intermediate work, but correct contradictions or mistakes. "
        "You must FINALIZE NOW, without further analysis or outline.\n\n"
        "USER REQUEST:\n" + user_text + "\n\n"
        "PARTIAL REASONING (may be mistaken):\n" + scratchpad[-3000:] + "\n\n"
        + format_instructions[answer_format]
    )

def load(model_name, device):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(2)
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends,"mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    if device=="mps":
        dtype=torch.float16
    elif device=="cuda":
        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype=torch.float32
    tokenizer=AutoTokenizer.from_pretrained(model_name)
    model=AutoModelForCausalLM.from_pretrained(
        model_name,torch_dtype=dtype,low_cpu_mem_usage=True).to(device).eval()
    return model, tokenizer, device

def generate(model, tok, device, messages, thinking, max_tokens, seed):
    import torch
    torch.manual_seed(seed)
    ids=tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,
                                enable_thinking=thinking,return_tensors="pt").to(device)
    with torch.inference_mode():
        out=model.generate(ids,attention_mask=torch.ones_like(ids),
                           do_sample=True,temperature=0.6 if thinking else 0.7,
                           top_p=0.95 if thinking else 0.8,top_k=20,
                           max_new_tokens=max_tokens,pad_token_id=tok.eos_token_id)
    raw=tok.decode(out[0,ids.shape[1]:],skip_special_tokens=False)
    return raw,int(ids.shape[1]),int(out.shape[1]-ids.shape[1])

def run(prompt,model,tok,device,thinking_budget,final_budget,answer_format,seed,compare):
    started=time.monotonic()
    messages=[{"role":"user","content":prompt}]
    baseline=None
    base_input=base_output=0
    if compare:
        baseline_raw,base_input,base_output=generate(
            model,tok,device,messages,False,160,seed+1)
        baseline=strip_end(baseline_raw)
    raw,initial_input,initial_output=generate(
        model,tok,device,messages,True,thinking_budget,seed+2)
    result,unfinished,scratchpad=interpret_thinking(raw)
    finish_input=finish_output=0
    finish_raw=None
    if unfinished:
        finish_raw,finish_input,finish_output=generate(
            model,tok,device,[
               {"role":"system","content":"Return the final answer immediately. Do not show reasoning."},
               {"role":"user","content":completion_prompt(prompt,scratchpad,answer_format)}
            ],False,final_budget,seed+3)
        result=strip_end(finish_raw)
    return dict(prompt=prompt,answer=result,forced_completion=unfinished,
                raw_thinking=raw,raw_forced_completion=finish_raw,
                comparison_plain=baseline,device=device,
                tokens=dict(plain_input=base_input,plain_output=base_output,
                            thinking_input=initial_input,thinking_output=initial_output,
                            finishing_input=finish_input,finishing_output=finish_output),
                elapsed_seconds=round(time.monotonic()-started,2))

def self_test():
    assert interpret_thinking("<think>2+3=5</think>FINAL ANSWER: 5<|im_end|>")==("FINAL ANSWER: 5",False,"2+3=5")
    assert interpret_thinking("<think>unfinished")[1]
    assert interpret_thinking("DIRECT REPLY<|im_end|>")[0]=="DIRECT REPLY"
    assert "FINALIZE NOW" in completion_prompt("q","scratch","text")
    assert "Python" in completion_prompt("q","scratch","python")
    print("PASS: generic Qwen3 final-answer detection and continuation")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("prompt",nargs="?",help="User request; omit to read from stdin")
    p.add_argument("--model",default="Qwen/Qwen3-0.6B",
                   help="Hugging Face model ID or local checkpoint path")
    p.add_argument("--device",choices=["auto","cpu","mps","cuda"],default="auto")
    p.add_argument("--thinking-tokens",type=int,default=512)
    p.add_argument("--final-tokens",type=int,default=192)
    p.add_argument("--answer-format",choices=["text","number","choice","python"],default="text")
    p.add_argument("--seed",type=int,default=20261114)
    p.add_argument("--compare",action="store_true",
                   help="Also generate an ordinary non-thinking answer for comparison")
    p.add_argument("--json-output",help="Write full trace and compute costs to this file")
    p.add_argument("--self-test",action="store_true")
    args=p.parse_args()
    if args.self_test:
        self_test()
        return
    prompt=args.prompt or sys.stdin.read().strip()
    if not prompt:
        p.error("Provide a prompt or pipe it through stdin")
    if args.thinking_tokens<=0 or args.final_tokens<=0:
        p.error("Token budgets must be positive")
    model,tok,device=load(args.model,args.device)
    output=run(prompt,model,tok,device,args.thinking_tokens,args.final_tokens,
               args.answer_format,args.seed,args.compare)
    print(output["answer"])
    if args.json_output:
        Path(args.json_output).write_text(json.dumps(output,indent=2))
        print("Trace:",args.json_output,file=sys.stderr)
    print("Generated tokens:",output["tokens"],file=sys.stderr)

if __name__=="__main__":
    main()
