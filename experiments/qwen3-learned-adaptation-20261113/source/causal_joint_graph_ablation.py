import torch,json
from pathlib import Path
P=Path('/mnt/data/qwen_next')
s=(P/'qwen_joint_graph_halt.py').read_text().split('net=Joint();')[0]
exec(compile(s,str(P/'qwen_joint_graph_halt.py'),'exec'))
net=Joint();net.load_state_dict(torch.load(P/'qwen_joint_graph_halt_curriculum/joint_graph_halt_curriculum_seed61426.pt',weights_only=True));net.eval()

def pred(data,mode):
 n=0;correct=0;vec=[]
 with torch.no_grad():
  for ix in torch.arange(len(data[2])).split(240):
   e,q,y,d=[v[ix] for v in data]
   if mode=='swap_story':e=torch.roll(e,3 if len(ix)%3==0 else 1,dims=0)
   if mode=='constant_story':e=e[:1].expand_as(e)
   z,*_=net(e,q)
   p=z.argmax(-1);v=(p==y);correct+=int(v.sum());n+=len(y);vec+=v.int().tolist()
 return {'correct':correct,'total':n,'vector':vec}
result={}
for key,data in [('held_out_1to3',te),('unseen_4',t4)]:
 result[key]={mode:pred(data,mode) for mode in ('intact','swap_story','constant_story')}
 print(key,{k:v['correct'] for k,v in result[key].items()},flush=True)
(P/'qwen_joint_graph_halt_curriculum'/'causal_state_ablation.json').write_text(json.dumps(result,indent=2))
