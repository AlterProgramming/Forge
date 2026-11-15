#!/usr/bin/env python3
"""Actual Qwen open-ended source reading and code editing. No prewritten patches."""
import argparse
import hashlib
import json
import re
import subprocess
import tempfile
from pathlib import Path
from qwen_recovery_smoke import ADAPTER,verify_local_checkpoint

SOURCE=Path("deliveries/forge-arena/final/index.html")
GOAL="""Make meaningful progress on the customer's substantial Ancient Greece
vehicle-combat game. The prior delivery added only Pause and the customer
rejected it. Read the existing source and write a useful, genuinely new feature
or substantial experience improvement. You decide what to change, not the harness.
Do not call the entire game finished without independent acceptance."""
REQUIREMENTS=[
    "The original outcome is a substantial enjoyable game, not a demo.",
    "The model must investigate and author its own new source code.",
    "Kairos must apply the model's actual proposed patch; Production Studio and Chromium validate it.",
    "Preserve prompts, unmodified user goal, actual tool actions, failures and separate gate outcomes."
]
SYSTEM="""You are a coding agent. You can read and edit the existing HTML game.
The CUSTOMER OBJECTIVE and INSTRUCTIONS are NOT game source and must NEVER be used
as edit anchors. Only exact strings from index.html can be edited.
If edit fails because find occurs 0 times, do not repeat the same edit.
Instead invoke read(query) on an actual source symbol and construct a new code edit.
Reply with ONE JSON object each turn. Tools:
{"tool":"read","query":"search term from the source"}
{"tool":"edit","rationale":"why it matters","edits":[{"find":"exact source text","replace":"new source text YOU wrote"}]}
{"tool":"finish","summary":"what you actually changed"}
Read returns source around the matching query. Every find must match exactly once.
Make real code changes to improve gameplay or presentation. Do not repeat Pause.
Do not claim code was changed until the edit tool verifies it. If rejected,
read the diagnostic and try again. There is no prepared code or action menu."""
def sha(s):return hashlib.sha256(s.encode("utf-8")).hexdigest()
def save(out,name,obj):(out/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False)+"\n")
def parse(raw):
    candidates=[raw]
    a,b=raw.find("{"),raw.rfind("}")
    if a>=0 and b>a:candidates.append(raw[a:b+1])
    for candidate in candidates:
        try:
            value=json.loads(candidate)
            if isinstance(value,dict) and value.get("tool") in ("read","edit","finish"):return value
        except json.JSONDecodeError:pass
    raise ValueError("No parseable tool call in generated text")
def check(source):
    scripts=re.findall(r"<script\b[^>]*>(.*?)</script>",source,re.S|re.I)
    if not scripts:raise ValueError("No embedded JavaScript")
    with tempfile.TemporaryDirectory() as folder:
        file=Path(folder)/"game.js";file.write_text("\n".join(scripts))
        result=subprocess.run(["node","--check",str(file)],capture_output=True,text=True,timeout=25)
        if result.returncode:raise ValueError("JS syntax failure: "+result.stderr[-900:])
    for value in ("function begin()","function pulse()", 'id="garage"', 'id="pause-btn"'):
        if value not in source:raise ValueError("Removed existing game component: "+value)
def edit(source,edits):
    if not isinstance(edits,list) or not 1<=len(edits)<=5:raise ValueError("1-5 model-written edits required")
    for item in edits:
        old,new=item.get("find"),item.get("replace")
        if not isinstance(old,str) or not isinstance(new,str) or not old or old==new:
            raise ValueError("find/replace must be different nonempty strings")
        if len(old)>5000 or len(new)>12000:raise ValueError("Edit exceeds bounded size")
        count=source.count(old)
        if count!=1:raise ValueError("Find must appear once; actual count "+str(count)+": "+repr(old[:90]))
        source=source.replace(old,new,1)
    check(source)
    return source
def read(source,query):
    if not isinstance(query,str) or not 2<=len(query)<=120:return "Invalid source query"
    pos=source.lower().find(query.lower())
    if pos<0:return "Source term not found: "+query
    return json.dumps({"offset":pos,"total":len(source),
      "source":source[max(0,pos-1800):min(len(source),pos+2700)]},ensure_ascii=False)
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out-dir",default="open-agent-evidence")
    parser.add_argument("--turns",type=int,default=7)
    args=parser.parse_args()
    out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
    original=SOURCE.read_text(encoding="utf-8")
    check(original)
    lora_sha=verify_local_checkpoint()
    save(out,"gate0.json",{"status":"PASS","source_sha256":sha(original),
        "lora_sha256":lora_sha,"instructions":REQUIREMENTS,"goal":GOAL,
        "code_generation_mode":"base Qwen3-0.6B with unrelated communication LoRA disabled"})
    print("OPEN_QWEN_GATE0",sha(original),flush=True)
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from peft import PeftModel
    torch.set_num_threads(2)
    tokenizer=AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    base=AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B",
        dtype=torch.float32,low_cpu_mem_usage=True).eval()
    model=PeftModel.from_pretrained(base,str(ADAPTER),is_trainable=False).eval()
    messages=[{"role":"system","content":SYSTEM},
      {"role":"user","content":"CUSTOMER OBJECTIVE:\n"+GOAL+
      "\nPRESERVED REQUIREMENTS:\n"+json.dumps(REQUIREMENTS)+
      "\nREAL GAME SOURCE SHA256 "+sha(original)+
      "\nSOURCE BEGINNING:\n"+original[:4400]+
      "\nSOURCE END:\n"+original[-6700:]+
      "\nYou can inspect the rest using read. Produce a meaningful original code improvement."}]
    source=original
    trace=[];patches=[]
    attempted_edits=set()
    for turn in range(args.turns):
        inputs=tokenizer.apply_chat_template(messages,tokenize=True,
            add_generation_prompt=True,enable_thinking=False,return_tensors="pt")
        with torch.inference_mode(),model.disable_adapter():
            outputs=model.generate(inputs,attention_mask=torch.ones_like(inputs),
                do_sample=False,max_new_tokens=1000,pad_token_id=tokenizer.eos_token_id)
        raw=tokenizer.decode(outputs[0,inputs.shape[1]:],skip_special_tokens=True).strip()
        row={"turn":turn,"input_sha256":sha(tokenizer.decode(inputs[0],skip_special_tokens=False)),
             "input_tokens":int(inputs.shape[1]),"raw_qwen_output":raw}
        trace.append(row);save(out,"model_trace.json",trace)
        print("OPEN_QWEN_RAW",json.dumps(row,ensure_ascii=False)[:9000],flush=True)
        messages.append({"role":"assistant","content":raw})
        try:
            action=parse(raw);row["tool"]=action["tool"]
            if action["tool"]=="read":
                observation=read(source,action.get("query"))
            elif action["tool"]=="edit":
                proposal_key=sha(json.dumps(action.get("edits"),sort_keys=True,ensure_ascii=False))
                if proposal_key in attempted_edits:
                    raise ValueError("REPEATED_IDENTICAL_EDIT: The same proposal already failed or was submitted. Choose a different tool action such as read(query), referencing actual index.html source.")
                attempted_edits.add(proposal_key)
                candidate=edit(source,action.get("edits"))
                patches.append({"turn":turn,"model_written_edit":action,
                    "before_sha256":sha(source),"after_sha256":sha(candidate)})
                source=candidate
                (out/"qwen_generated_game.html").write_text(source,encoding="utf-8")
                save(out,"model_generated_patches.json",patches)
                observation="EDIT APPLIED to real game; node --check passed; SHA256 "+sha(source)
            else:
                observation="Finished after "+str(len(patches))+" verified edits" if patches else "Cannot finish without writing code"
                if patches:
                    row["tool_result"]=observation
                    break
        except Exception as exc:
            observation="TOOL REJECTED: "+str(exc)
            if "Find must appear once; actual count 0" in observation:
                observation += " The requested FIND is absent from index.html. The CUSTOMER OBJECTIVE is a prompt, not source. Use read(query) for a real game function such as begin or pulse before editing."
        row["tool_result"]=observation
        save(out,"model_trace.json",trace)
        print("OPEN_QWEN_TOOL",json.dumps({"turn":turn,"result":observation[:700]}),flush=True)
        messages.append({"role":"user","content":"ACTUAL TOOL RESULT: "+observation+
            "\nOriginal customer objective is unchanged. Supply next tool JSON:"})
    status="PASS" if patches else "FAIL"
    save(out,"gate1.json",{"status":status,"gate":"Qwen writes new code","source_sha256":sha(original),
        "result_sha256":sha(source),"model_authored_edits":len(patches),"turns":len(trace),
        "parent_game_delivery":"NOT ACCEPTED; separate product assessment required"})
    print("OPEN_QWEN_GATE1",status,"model edits:",len(patches),flush=True)
    if not patches:raise SystemExit(2)
if __name__=="__main__":main()
