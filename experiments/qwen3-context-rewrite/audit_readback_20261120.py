#!/usr/bin/env python3
"""Reconstruct the fixed test labels and evaluate frozen-Qwen run scores."""
import random, json
from collections import Counter
from math import comb
scores={"parity/stateful":[13,13,11],"parity/blind":[13,13,13],
"last_write/stateful":[12,11,13],"last_write/blind":[9,13,13]}
out={"source_run":"https://github.com/AlterProgramming/Forge/actions/runs/44637769586","scores_len3_len6_ablate_len6":scores,"datasets":{},"statistics":{}}
for task in ("parity","last_write"):
 for length in (3,6):
  rng=random.Random(10000+length)
  xs=[tuple(rng.randrange(2) for _ in range(length)) for _ in range(24)]
  ys=[sum(x)%2 if task=="parity" else x[-1] for x in xs]
  out["datasets"][f"{task}/{length}"]={"label_counts":dict(Counter(ys)),"distinct_inputs":len(set(xs)),"constant_majority_correct":max(Counter(ys).values())}
for key,(a,b,c) in scores.items():
 out["statistics"][key]={"len6_accuracy":b/24,"len6_ablation_delta_correct":b-c,"two_sided_binomial_p_vs_half":min(1,2*sum(comb(24,i) for i in range(max(b,24-b),25))/2**24)}
print(json.dumps(out,indent=2))
