#!/usr/bin/env python3
"""Follow-up read-argument grounding: Qwen native function calls vs source-window size.

Read-only, no candidate solution code. Uses the same real game source SHA as
the prior nine-condition experiment, but different prompt formulations.
"""
import json
from pathlib import Path
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from probe import source,READ_TOOL,digest,parse_result
from time import monotonic

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"tool-window-evidence"
OUT.mkdir(exist_ok=True)
src=source()
idx=src.index("function begin()")
tokenizer=AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
torch.set_num_threads(2)
model=AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B",
    dtype=torch.float32,low_cpu_mem_usage=True).eval()
cases=[
 {"id":"B1-native-literal-compact","kind":"native","sample":True,"seed":42,"span":220,
  "system":"Call read_source with an EXACT substring from source text, not a description of the task. Its query MUST be literal characters visible in the SOURCE. A search for natural-language intent will fail.",
  "task":"Locate the function that starts a game battle."},
 {"id":"B2-native-literal-medium","kind":"native","sample":True,"seed":42,"span":1500,
  "system":"Call read_source with an EXACT substring from source text, not a description of the task. Its query MUST be literal characters visible in the SOURCE. A search for natural-language intent will fail.",
  "task":"Locate the function that starts a game battle."},
 {"id":"B3-native-literal-long","kind":"native","sample":True,"seed":42,"span":4000,
  "system":"Call read_source with an EXACT substring from source text, not a description of the task. Its query MUST be literal characters visible in the SOURCE. A search for natural-language intent will fail.",
  "task":"Locate the function that starts a game battle."},
 {"id":"B4-native-other-seed","kind":"native","sample":True,"seed":91,"span":220,
  "system":"Call read_source with an EXACT substring from source text, not a description of the task. Its query MUST be literal characters visible in the SOURCE. A search for natural-language intent will fail.",
  "task":"Locate the function that starts a game battle."},
 {"id":"B5-native-with-good-example","kind":"native","sample":True,"seed":42,"span":220,
  "system":"Call read_source with a literal source string. For example, when reading a function named pulse, a correct query is function pulse(). The task now involves a DIFFERENT function.",
  "task":"Locate the function that starts a game battle."},
 {"id":"B6-json-different-seed","kind":"json","sample":True,"seed":91,"span":220,
  "system":"Return one JSON object with tool=read and query equal to actual code characters that occur in the source. Do not copy placeholder examples.",
  "task":"Locate the function that starts a game battle."},
 {"id":"B7-plain-source-query","kind":"text","sample":True,"seed":42,"span":1500,
  "system":"Respond ONLY with a unique searchable substring copied from the actual source identifying the battle-start function. No explanation and no extra wrapper.",
  "task":"Locate the function that starts a game battle."},
]
results=[]
for c in cases:
    torch.manual_seed(c["seed"])
    snippet=src[idx:idx+c["span"]]
    messages=[{"role":"system","content":c["system"]},
      {"role":"user","content":c["task"]+"\nSOURCE (actual game file):\n"+snippet}]
    kwargs={"tokenize":True,"add_generation_prompt":True,
       "enable_thinking":False,"return_tensors":"pt"}
    if c["kind"]=="native":kwargs["tools"]=[READ_TOOL]
    ids=tokenizer.apply_chat_template(messages,**kwargs)
    t=monotonic()
    with torch.inference_mode():
        generated=model.generate(ids,attention_mask=torch.ones_like(ids),
            do_sample=True,temperature=0.7,top_p=0.8,top_k=20,
            max_new_tokens=90,pad_token_id=tokenizer.eos_token_id)
    raw=tokenizer.decode(generated[0,ids.shape[1]:],skip_special_tokens=True).strip()
    query=parse_result(raw,c["kind"])
    record={**c,"source_sha256":digest(src),"excerpt":snippet,
      "prompt_sha256":digest(tokenizer.decode(ids[0],skip_special_tokens=False)),
      "input_tokens":int(ids.shape[1]),"generated_tokens":int(generated.shape[1]-ids.shape[1]),
      "seconds":round(monotonic()-t,2),"raw_output":raw,"query":query,
      "query_occurs_in_actual_source":bool(query and query in src),
      "exact_target":query=="function begin()",
      "query_occurs_in_visible_excerpt":bool(query and query in snippet)}
    results.append(record)
    (OUT/"results.json").write_text(json.dumps(results,indent=2,ensure_ascii=False)+"\n")
    print("WINDOW_PROBE",json.dumps({k:record[k] for k in (
      "id","query","query_occurs_in_actual_source",
      "query_occurs_in_visible_excerpt","exact_target","generated_tokens","seconds")},ensure_ascii=False),flush=True)
report={"total":len(results),"valid_substrings":sum(x["query_occurs_in_actual_source"] for x in results),
    "native_trials":sum(x["kind"]=="native" for x in results),
    "native_hits":sum(x["kind"]=="native" and x["query_occurs_in_actual_source"] for x in results),
    "source_sha256":digest(src),
    "warning":"Tiny diagnostic examples; not general tool competence or full coding ability."}
(OUT/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
print("WINDOW_SUMMARY",json.dumps(report),flush=True)
