#!/usr/bin/env python3
"""Posthoc causal probe of Qwen3 code errors, with independent held-out tests.

Compare same frozen Qwen's deterministic answer before and after (A) supplying
a semantic invariant, and (B) supplying one executable failing assertion and
its own code for repair. All evaluation uses OTHER public MBPP test cases.
The test shown to model is development supervision: results do not count as
generalization to unseen tasks. This is not safe to run with arbitrary code
outside a disposable isolated runner.
"""
import argparse
import json
from pathlib import Path
import time
from boost import load_tasks, python_code, run_tests

CASES={
 "433": ("quantifier and variable shadowing", [
    "Keep the input number distinct from each loop element; do not shadow parameters.",
    "The test should be true iff the entered number is strictly greater than EVERY element of the array.",
    "Match the function name and positional input order in the user's request."
 ]),
 "246": ("iteration invariant and invalid initialization", [
    "Use Babylonian/Newton iteration for x*x=a: next guess = (guess + a/guess)/2.",
    "Start from a NONZERO guess; handle a=0 explicitly.",
    "Stop based on convergence between successive guesses, not on guess=a."
 ]),
 "160": ("Diophantine constraints versus arbitrary placeholders", [
    "You need integer x and y with a*x + b*y == n, not arbitrary chosen values.",
    "Use the extended Euclidean algorithm, checking whether gcd(a,b) divides n.",
    "Return values that actually satisfy the equation; do not return uninitialized variables."
 ])
}
SYSTEM="Return only complete runnable Python code. No explanatory prose."

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model",default="Qwen/Qwen3-0.6B")
    p.add_argument("--output",default="code_causal_results.json")
    p.add_argument("--self-test",action="store_true")
    args=p.parse_args()
    if args.self_test:
        assert len(CASES)==3
        assert all(c[0] and c[1] for c in CASES.values())
        print("PASS code-case annotations")
        return
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    torch.set_num_threads(2)
    all_tasks=load_tasks(6,20261115)
    selected={t["id"]:t for t in all_tasks if t["family"]=="mbpp" and t["id"] in CASES}
    if set(selected)!=set(CASES):
        raise ValueError(f"Dataset task mismatch: {set(selected)}")
    tok=AutoTokenizer.from_pretrained(args.model)
    model=AutoModelForCausalLM.from_pretrained(args.model,dtype=torch.float32,
                                              low_cpu_mem_usage=True).eval()
    device=next(model.parameters()).device
    def ask(prompt):
        messages=[{"role":"system","content":SYSTEM},{"role":"user","content":prompt}]
        ids=tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,
                                    enable_thinking=False,return_tensors="pt").to(device)
        with torch.inference_mode():
            y=model.generate(ids,attention_mask=torch.ones_like(ids),do_sample=False,
                             max_new_tokens=240,pad_token_id=tok.eos_token_id)
        raw=tok.decode(y[0,ids.shape[1]:],skip_special_tokens=True)
        return dict(raw=raw,code=python_code(raw),input_tokens=int(ids.shape[1]),
                    output_tokens=int(y.shape[1]-ids.shape[1]))
    start=time.monotonic()
    rows=[]
    for key,(kind,hints) in CASES.items():
        t=selected[key]
        original=ask(t["prompt"])
        anchored=ask(t["prompt"]+"\n\nIMPORTANT REQUIREMENTS:\n" +
                     "\n".join("- "+x for x in hints))
        first_dev=t["dev_tests"][0]
        debugging=ask(t["prompt"]+"\n\nYOUR PREVIOUS CANDIDATE:\n"+
                      original["code"]+
                      "\n\nIt must pass this single PUBLIC DEVELOPMENT TEST:\n"+
                      first_dev+
                      "\n\nRepair any semantic or runtime error. Return fully corrected runnable Python code.")
        answers={"original":original,"invariant_only":anchored,"one_test_repair":debugging}
        for label,a in answers.items():
            a["dev_pass"]=run_tests(a["code"],t["dev_tests"],t["setup"])
            a["heldout_pass"]=run_tests(a["code"],t["eval_tests"],t["setup"])
            print(f'{key} {label}: dev={a["dev_pass"]} heldout={a["heldout_pass"]}',flush=True)
        row=dict(id=key,failure_kind=kind,task=t["prompt"],
                 verifier_development_example=first_dev,semantic_invariant=hints,
                 arms=answers)
        rows.append(row)
        report=dict(protocol="posthoc code diagnosis on previously examined MBPP tasks",
                    model=args.model,rows=rows,
                    summary={arm:sum(int(r["arms"][arm]["heldout_pass"]) for r in rows)
                             for arm in answers},
                    elapsed_seconds=round(time.monotonic()-start,2))
        Path(args.output).write_text(json.dumps(report,indent=2))
    print(json.dumps(report["summary"],indent=2),flush=True)

if __name__=="__main__":main()
