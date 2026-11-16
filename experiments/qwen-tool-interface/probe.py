#!/usr/bin/env python3
"""Controlled, read-only Qwen3-0.6B tool-interface experiment.

No prewritten model choices, no edits, no source mutations.
Compare prompt example leakage, plain JSON, official Qwen function schemas,
and bounded feedback with short independent prompts and verified source hits.
Each model response is preserved verbatim; all outcomes are reported.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path
from time import monotonic

ROOT=Path(__file__).resolve().parents[2]
GAME=ROOT/"deliveries/forge-arena/source/index.html"
OUT=ROOT/"tool-probe-evidence"
READ_TOOL={
 "type":"function",
 "function":{
  "name":"read_source",
  "description":"Search the actual existing game source for an exact text phrase and return nearby source. Invent the phrase from the file, not from example syntax.",
  "parameters":{
    "type":"object",
    "properties":{"query":{"type":"string","description":"An actual substring copied from the SOURCE excerpt."}},
    "required":["query"]
  }
 }
}
def digest(s):return hashlib.sha256(s.encode()).hexdigest()
def store(name,object):
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/name).write_text(json.dumps(object,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")

def source():
    s=GAME.read_text(encoding="utf-8")
    anchors=["function begin()","function pulse()","id=\"garage\"","function draw()"]
    for a in anchors:
        assert a in s, "Ground truth missing "+a
    return s

def cases(s):
    # Actual code excerpts, not pseudo source invented for a positive outcome.
    begin=s.index("function begin()")
    excerpt=s[begin:begin+400]
    tiny="EXISTING GAME SOURCE:\n"+excerpt
    return [
      {"id":"A1-old-example-trap","kind":"json","sample":False,
       "system":"You can inspect the game with JSON tools. Reply exactly one JSON object. Tool: {\"tool\":\"read\",\"query\":\"search term from the source\"}. Replace example placeholder with the real source phrase.",
       "user":"Find the battle-start function in the source.\n"+tiny},
      {"id":"A2-typed-placeholder","kind":"json","sample":False,
       "system":"Return only a JSON object with keys tool and query. tool must be read. query must be an exact substring of SOURCE; the word QUERY_VALUE is a placeholder, not a source string. Never use QUERY_VALUE verbatim.",
       "user":"Find battle start function in this source:\n"+tiny},
      {"id":"A3-no-example","kind":"json","sample":False,
       "system":"You are a code inspection assistant. Reply with a JSON object naming a read operation and the exact text to find. Use keys tool and query, with tool set to read.",
       "user":"Which actual source string should be searched to locate the battle-start function?\n"+tiny},
      {"id":"A4-valid-example","kind":"json","sample":False,
       "system":"Inspect the source. Return a JSON read call with the exact source substring. A real valid call looks like {\"tool\":\"read\",\"query\":\"function pulse()\"}. Choose the relevant real source phrase rather than copying pulse blindly.",
       "user":"Find the battle-start function.\n"+tiny},
      {"id":"A5-plain-natural","kind":"text","sample":False,
       "system":"Inspect source. Reply with only the EXACT substring (no JSON, quotes, or commentary) that locates the battle start function.",
       "user":tiny},
      {"id":"A6-official-tools-greedy","kind":"native","sample":False,
       "system":"You have a function for inspecting the existing code. Call it with a real searchable phrase appearing in SOURCE.",
       "user":"Find the battle start implementation. Use read_source on an exact phrase from this real game excerpt:\n"+excerpt},
      {"id":"A7-official-tools-sampled","kind":"native","sample":True,
       "system":"You have a function for inspecting the existing code. Call it with a real searchable phrase appearing in SOURCE.",
       "user":"Find the battle start implementation. Use read_source on an exact phrase from this real game excerpt:\n"+excerpt},
      {"id":"A8-json-sampled","kind":"json","sample":True,
       "system":"Return one JSON object with tool=read and query equal to actual code characters that occur in the source. Do not copy placeholder examples.",
       "user":"Find the battle-start function.\n"+tiny},
      {"id":"A9-natural-sampled","kind":"text","sample":True,
       "system":"Reply only with the actual exact source substring that identifies the battle-start function.",
       "user":tiny},
    ]

def parse_result(raw,kind):
    if kind=="text":
        return raw.strip().strip('"\u0060 ')
    parts=[]
    if kind=="native":
        parts+=re.findall(r"<tool_call>(.*?)</tool_call>",raw,re.S)
    parts.append(raw)
    a,b=raw.find("{"),raw.rfind("}")
    if a>=0 and b>a:parts.append(raw[a:b+1])
    for part in parts:
        try:
            obj=json.loads(part.strip())
        except (ValueError,TypeError):
            continue
        if not isinstance(obj,dict):continue
        if kind=="native":
            args=obj.get("arguments",{})
            if isinstance(args,str):
                try:args=json.loads(args)
                except ValueError:continue
            if isinstance(args,dict) and isinstance(args.get("query"),str):
                return args["query"]
        elif isinstance(obj.get("query"),str):
            return obj["query"]
    return None

def infer(model,tokenizer,messages,kind,sampled,max_new_tokens=120):
    import torch
    template_args={"tokenize":True,"add_generation_prompt":True,
        "enable_thinking":False,"return_tensors":"pt"}
    if kind=="native":template_args["tools"]=[READ_TOOL]
    ids=tokenizer.apply_chat_template(messages,**template_args)
    start=monotonic()
    options={"max_new_tokens":max_new_tokens,"pad_token_id":tokenizer.eos_token_id}
    if sampled:
        options.update({"do_sample":True,"temperature":0.7,"top_p":0.8,"top_k":20})
    else:
        options.update({"do_sample":False})
    with torch.inference_mode():
        gen=model.generate(ids,attention_mask=torch.ones_like(ids),**options)
    raw=tokenizer.decode(gen[0,ids.shape[1]:],skip_special_tokens=True).strip()
    return {
      "raw_output":raw,
      "input_tokens":int(ids.shape[1]),
      "generated_tokens":int(gen.shape[1]-ids.shape[1]),
      "prompt_sha256":digest(tokenizer.decode(ids[0],skip_special_tokens=False)),
      "seconds":round(monotonic()-start,2),
      "truncated":int(gen.shape[1]-ids.shape[1])>=max_new_tokens
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--limit",type=int,default=9)
    args=ap.parse_args()
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    torch.set_num_threads(2)
    src=source()
    store("setup.json",{"model":"Qwen/Qwen3-0.6B","base_only":True,
        "communication_lora":"not applied (irrelevant for source-reading tests)",
        "model_mode":"non-thinking","game_source_sha256":digest(src),
        "source_path":str(GAME),"samples":["greedy","sampled"],
        "official_tool_schema":READ_TOOL,
        "task":"Find existing function begin() without copying prompt placeholders"})
    tokenizer=AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    model=AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B",
        dtype=torch.float32,low_cpu_mem_usage=True).eval()
    results=[]
    for case in cases(src)[:args.limit]:
        messages=[{"role":"system","content":case["system"]},
                  {"role":"user","content":case["user"]}]
        result=infer(model,tokenizer,messages,case["kind"],case["sample"])
        query=parse_result(result["raw_output"],case["kind"])
        valid=bool(query) and query in src
        result.update({"id":case["id"],"kind":case["kind"],"sampled":case["sample"],
            "system":case["system"],"user":case["user"],
            "parsed_query":query,"actual_source_hit":valid,
            "selected_target":query=="function begin()",
            "copied_placeholder":bool(query and query in (
              "search term from the source","QUERY_VALUE","exact source text")),
            "not_just_source_grounding":"Exact target is begin(); other real source strings count as hits but not target selection"})
        results.append(result)
        store("results.json",results)
        print("INTERFACE_PROBE",json.dumps({
            "id":result["id"],"query":query,"hit":valid,
            "target":result["selected_target"],"placeholder":result["copied_placeholder"],
            "tokens":result["generated_tokens"],"elapsed_s":result["seconds"],
            "truncated":result["truncated"],"raw":result["raw_output"][:900]},ensure_ascii=False),flush=True)
    # Feedback recovery, in a new isolated conversation; not the previous
    # source transcript. The ground truth is not supplied as answer.
    case=cases(src)[0]
    messages=[{"role":"system","content":case["system"]},{"role":"user","content":case["user"]}]
    first=infer(model,tokenizer,messages,case["kind"],False,max_new_tokens=90)
    q1=parse_result(first["raw_output"],"json")
    msg=("Actual tool error: query "+repr(q1)+" did not occur in the source. "
         "The placeholder from the tool definition is not code. "
         "Choose a distinct phrase that really occurs in the SOURCE excerpt.")
    messages += [{"role":"assistant","content":first["raw_output"]},
                 {"role":"user","content":msg}]
    second=infer(model,tokenizer,messages,case["kind"],True,max_new_tokens=90)
    q2=parse_result(second["raw_output"],"json")
    recovery={"scenario":"fresh invalid-query feedback, two-turn",
      "first":first,"first_query":q1,"first_hit":bool(q1 and q1 in src),
      "feedback":msg,"second":second,"second_query":q2,
      "second_hit":bool(q2 and q2 in src),"changed":q1!=q2}
    store("feedback-recovery.json",recovery)
    print("FEEDBACK_RECOVERY",json.dumps({"first_query":q1,"second_query":q2,
          "first_hit":recovery["first_hit"],"second_hit":recovery["second_hit"],
          "changed":recovery["changed"]},ensure_ascii=False),flush=True)
    summary={"total_conditions":len(results),
        "source_hit_count":sum(int(x["actual_source_hit"]) for x in results),
        "exact_target_count":sum(int(x["selected_target"]) for x in results),
        "placeholder_count":sum(int(x["copied_placeholder"]) for x in results),
        "feedback_recovery":recovery["second_hit"],
        "note":"Small, diagnostic prompt sensitivity on one source file, NOT proof of general model capabilities."}
    store("summary.json",summary)
    print("INTERFACE_SUMMARY",json.dumps(summary),flush=True)

if __name__=="__main__":
    main()
