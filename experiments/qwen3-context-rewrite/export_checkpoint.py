#!/usr/bin/env python3
"""Package public Qwen3-0.6B checkpoint in verifiable small upload artifacts."""
import hashlib, json, os
from pathlib import Path
from huggingface_hub import snapshot_download

DEST=Path("transfer")
MODEL_DIR=DEST/"model"
SHARDS=DEST/"shards"
CHUNK_SIZE=192*1024*1024

MODEL_DIR.mkdir(parents=True,exist_ok=True)
SHARDS.mkdir(parents=True,exist_ok=True)
snapshot_download(
 repo_id="Qwen/Qwen3-0.6B",
 local_dir=str(MODEL_DIR),
 allow_patterns=["*.safetensors","*.json","*.txt","*.model"],
)
manifest={"model":"Qwen/Qwen3-0.6B","chunk_size":CHUNK_SIZE,"files":[],"parts":[]}
number=0
for item in sorted(MODEL_DIR.rglob("*")):
 if not item.is_file():continue
 relative=item.relative_to(MODEL_DIR).as_posix()
 sha=hashlib.sha256()
 pieces=[]
 with item.open("rb") as fh:
  while True:
   blob=fh.read(CHUNK_SIZE)
   if not blob:break
   sha.update(blob)
   part_name=f"part-{number:03d}"
   (SHARDS/part_name).write_bytes(blob)
   pieces.append(part_name)
   manifest["parts"].append({"name":part_name,"size":len(blob),"sha256":hashlib.sha256(blob).hexdigest()})
   number+=1
 manifest["files"].append({"name":relative,"size":item.stat().st_size,"sha256":sha.hexdigest(),"parts":pieces})
(DEST/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
print("PACKED",len(manifest["files"]),"files into",number,"parts",flush=True)
for f in manifest["files"]:print(f["name"],f["size"],len(f["parts"]),flush=True)
