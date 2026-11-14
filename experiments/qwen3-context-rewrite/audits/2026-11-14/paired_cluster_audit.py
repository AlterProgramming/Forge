import json,random,collections,math,time,sys
from pathlib import Path
p=Path(__file__).parent/'qwen3-learned-adaptation/results/qwen_joint_graph_halt_curriculum'
N=1000
out={'experiment':'story-cluster paired bootstrap and within-story sign permutation','bootstrap_replicates':N,'seed':20261114,'source_files':[],'results':{}}
for eval_seed in (61425,61426):
 x=json.loads((p/f'paired_test_predictions_eval_seed{eval_seed}.json').read_text());out['source_files'].append(f'paired_test_predictions_eval_seed{eval_seed}.json')
 for split in ('test','ood'):
  ids=x[f'{split}_story_ids'];truth=x[f'{split}_ground_truth']; groups=collections.defaultdict(list)
  for i,s in enumerate(ids):groups[s].append(i)
  story=list(groups.values());rng=random.Random(20261114+eval_seed+(split=='ood'))
  methods=x['methods'];pairs=[('curriculum_seed61425','no_curriculum_seed61425'),('curriculum_seed61425','permuted_features_seed61425'),('curriculum_seed61426','permuted_features_seed61425')]
  for a,b in pairs:
   if a not in methods or b not in methods:continue
   ac=methods[a][f'correct_{split}'];bc=methods[b][f'correct_{split}']
   if isinstance(ac,int):ac=[int(y==t) for y,t in zip(methods[a][f'{split}_pred'],truth)]
   if isinstance(bc,int):bc=[int(y==t) for y,t in zip(methods[b][f'{split}_pred'],truth)]
   if len(ac)!=len(ids) or len(bc)!=len(ids):raise RuntimeError('alignment')
   deltas=[sum(ac[i]-bc[i] for i in g) for g in story];counts=[len(g) for g in story]
   obs=sum(deltas)/len(ids);samples=[];pos=neg=0
   for _ in range(N):
    chosen=[rng.randrange(len(story)) for __ in story];samples.append(sum(deltas[j] for j in chosen)/sum(counts[j] for j in chosen))
    null=sum(d*(1 if rng.getrandbits(1) else -1) for d in deltas)/len(ids)
    if abs(null)>=abs(obs)-1e-12:pos+=1
   samples.sort();key=f'{eval_seed}/{split}/{a}_vs_{b}'
   out['results'][key]={'n_questions':len(ids),'n_stories':len(story),'a_correct':sum(ac),'b_correct':sum(bc),'delta_accuracy':round(obs,6),'cluster_bootstrap_95ci':[round(samples[int(.025*N)],6),round(samples[int(.975*N)],6)],'story_signflip_two_sided_p':round((pos+1)/(N+1),6),'discordant_a_only':sum(v and not w for v,w in zip(ac,bc)),'discordant_b_only':sum(w and not v for v,w in zip(ac,bc))}
print(json.dumps(out,indent=2));(Path(__file__).parent/'paired_cluster_audit_results.json').write_text(json.dumps(out,indent=2)+'\n')