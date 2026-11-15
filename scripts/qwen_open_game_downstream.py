#!/usr/bin/env python3
"""Route *actual model-authored* game source through existing Kairos and Studio.

This module contains zero candidate patches. It replays the model's proposed
find/replace operations, then submits the model's exact result to Kairos.
Neither successful syntax checks nor screenshots imply complete user acceptance.
"""
import hashlib
import json
from pathlib import Path
import sys

from qwen_open_game_agent import SOURCE,edit,sha,check

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"open-agent-evidence"
def save(name,obj):
    (OUT/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=str)+"\n")
def main():
    gate=json.loads((OUT/"gate1.json").read_text())
    if gate["status"]!="PASS":raise RuntimeError("Model did not provide verified original code")
    original=SOURCE.read_text()
    patches=json.loads((OUT/"model_generated_patches.json").read_text())
    if not patches:raise RuntimeError("No real Qwen patches")
    replay=original
    for patch in patches:
        if sha(replay)!=patch["before_sha256"]:raise RuntimeError("Model proposal lineage mismatch")
        replay=edit(replay,patch["model_written_edit"]["edits"])
        if sha(replay)!=patch["after_sha256"]:raise RuntimeError("Model patch produces incorrect bytes")
    proposed=(OUT/"qwen_generated_game.html").read_text()
    if replay!=proposed or sha(replay)==sha(original):raise RuntimeError("Not model-authored changed source")
    (OUT/"model-authored-diff-summary.json").write_text(
        json.dumps({"base_sha256":sha(original),"model_sha256":sha(proposed),
                    "edits":len(patches),"generated_by":"Qwen3-0.6B, not hardcoded patch library"},indent=2)+"\n")
    check(proposed)
    sys.path.insert(0,str(ROOT/"kairos_repo"))
    from kairos.service.execution import execute_bounded_task
    test=(
      "from pathlib import Path\n"
      "def test_core_game_and_changed_source():\n"
      "    s=Path('index.html').read_text(encoding='utf-8')\n"
      "    assert 'function begin()' in s\n"
      "    assert 'function pulse()' in s\n"
      "    assert 'id=\"garage\"' in s\n"
      "    assert 'id=\"pause-btn\"' in s\n"
      f"    assert __import__('hashlib').sha256(s.encode()).hexdigest()=={sha(proposed)!r}\n"
    )
    payload={
      "request_id":"model-authored-game-retry",
      "request":"Improve the original customer-requested Ancient Greece vehicle game; keep full task open until acceptance",
      "objective":"Apply Qwen's own generated source patch, then check validity",
      "outcome":"Model-authored code is applied unchanged and validated; product quality remains separately assessed",
      "files":{"index.html":original,"test_game.py":test},
      "plan":{"comment":"Model-generated patch, not a preprogrammed transformation",
        "steps":[
          {"kind":"read_file","description":"Inspect actual original game","target_path":"index.html"},
          {"kind":"propose_patch","description":"Submit the exact code Qwen authored",
           "target_path":"index.html","proposed_content":proposed},
          {"kind":"apply_patch","description":"Apply Qwen's proposed code",
           "target_path":"index.html","proposed_content":proposed,"depends_on_index":1},
          {"kind":"run_validation","description":"Check game structural contract and Qwen bytes",
           "target_path":".","command":["python3","-m","pytest","test_game.py","-q"],"depends_on_index":2}
        ]},
      # This approval is a bounded-work execution authorization, *not* Kairos
      # independently judging code quality or end-user acceptance.
      "approval":{"approved":True,"decision_actor":"orchestrator-bounded-experiment",
        "authority_ref":"User authorized retry of code-generation experiment; review required before merge"}
    }
    receipt=execute_bounded_task(payload)
    save("gate2-kairos-receipt.json",receipt)
    if not receipt.get("success") or receipt.get("output_files",{}).get("index.html")!=proposed:
        raise RuntimeError("Kairos failed to apply exact model-authored code")
    print("OPEN_QWEN_KAIROS_PASS",sha(proposed),flush=True)

    sys.path.insert(0,str(ROOT/"studio_repo"))
    from studio_builder import StudioBuilder
    workspace=ROOT/"studio_workspace"
    workspace.mkdir(exist_ok=True)
    (workspace/"qwen-game.html").write_text(proposed)
    plan={"schema_version":"1.0","plan_id":"open-qwen-game-proposal-20261115",
      "allowed_paths":["output/**"],"preconditions":[],
      "operations":[{"id":"write-game","type":"copy_file","source":"qwen-game.html","target":"output/index.html"}],
      "postconditions":[{"id":"game-start","type":"text_contains","path":"output/index.html",
                          "value":"function begin()"},
                        {"id":"garage","type":"text_contains","path":"output/index.html",
                          "value":"id=\"garage\""}]}
    builder=StudioBuilder(workspace)
    preview=builder.preview(plan)
    studio=builder.apply(plan)
    verification=builder.verify_receipt(studio)
    save("gate3-studio-receipt.json",{"preview":preview,"studio":studio,"verification":verification})
    if verification["status"]!="PASS":raise RuntimeError("StudioBuilder output could not be verified")
    final=workspace/"output/index.html"
    if sha(final.read_text())!=sha(proposed):raise RuntimeError("Studio changed the model's source")
    (OUT/"kairos_studio_game.html").write_text(final.read_text())
    print("OPEN_QWEN_STUDIO_PASS",sha(final.read_text()),flush=True)
if __name__=="__main__":main()
