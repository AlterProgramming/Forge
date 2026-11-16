#!/usr/bin/env python3
"""Independent distractor-control tasks for Qwen tool-call source selection.

Four different real functions appear together in each prompt. The model must
select by a behavior description, not by a preselected isolated function.
Read-only; no tool payload examples with the answer; no mutations.
"""
import json
import torch
from pathlib import Path
from time import monotonic
from transformers import AutoTokenizer,AutoModelForCausalLM
from probe import source,READ_TOOL,digest,parse_result
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"tool-distractor-evidence"
OUT.mkdir(exist_ok=True)
src=source()
functions=[
    ("function begin()",360),
    ("function pulse()",360),
    ("function finish(win)",360),
    ("function spawnEnemy(i,boss)",360),
]
excerpts=[]
for anchor,length in functions:
    pos=src.index(anchor)
    excerpts.append(src[pos:pos+length])
deck="\n\nNEXT FUNCTION:\n".join(excerpts)
tasks=[
 {"id":"C1-wave-start","task":"Find the source function that initializes a new battle and its initial enemy wave.","target":"function begin()"},
 {"id":"C2-energy-ability","task":"Find the function that spends energy on the player's area ability.","target":"function pulse()"},
 {"id":"C3-end-rewards","task":"Find the function that gives rewards and advances progress after a battle ends.","target":"function finish(win)"},
 {"id":"C4-enemy-creation","task":"Find the function that constructs and adds new enemies to the current battle.","target":"function spawnEnemy(i,boss)"}
]
tok=AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
torch.set_num_threads(2)
net=AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B",dtype=torch.float32,low_cpu_mem_usage=True).eval()
results=[]
for k,c in enumerate(tasks):
    torch.manual_seed(102+k)
    messages=[
        {"role":"system","content":"You have read_source to inspect an existing source file. Call it with the EXACT literal source string identifying the function that best matches the requested behavior. Do not send descriptions or task text as the query. Only source substrings work."},
        {"role":"user","content":"Task: "+c["task"]+"\nThe same four real, distinct game functions are shown below. Search by function declaration:\n"+deck}
    ]
    ids=tok.apply_chat_template(messages,tools=[READ_TOOL],tokenize=True,
        add_generation_prompt=True,enable_thinking=False,return_tensors="pt")
    start=monotonic()
    with torch.inference_mode():
        gen=net.generate(ids,attention_mask=torch.ones_like(ids),do_sample=True,
            temperature=0.7,top_p=0.8,top_k=20,max_new_tokens=110,
            pad_token_id=tok.eos_token_id)
    raw=tok.decode(gen[0,ids.shape[1]:],skip_special_tokens=True).strip()
    q=parse_result(raw,"native")
    result=dict(**c,query=q,correct=q==c["target"],
       query_in_source=bool(q and q in src),query_in_visible_deck=bool(q and q in deck),
       raw_response=raw,input_tokens=int(ids.shape[1]),
       generated_tokens=int(gen.shape[1]-ids.shape[1]),
       seconds=round(monotonic()-start,2),
       source_sha256=digest(src),prompt_sha256=digest(tok.decode(ids[0],skip_special_tokens=False)))
    results.append(result)
    (OUT/"results.json").write_text(json.dumps(results,indent=2,ensure_ascii=False)+"\n")
    print("DISTRACTOR_PROBE",json.dumps({k:result[k] for k in (
      "id","query","target","correct","query_in_source","input_tokens","generated_tokens","seconds")},ensure_ascii=False),flush=True)
summary=dict(total=len(results),exact_target=sum(x["correct"] for x in results),
             source_grounded=sum(x["query_in_source"] for x in results),
             interpretation="Small independent function-discrimination test, not an autonomy benchmark")
(OUT/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
print("DISTRACTOR_SUMMARY",json.dumps(summary),flush=True)
