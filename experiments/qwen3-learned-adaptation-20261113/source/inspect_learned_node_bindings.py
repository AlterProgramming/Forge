import torch,re,json
from pathlib import Path
from scipy.optimize import linear_sum_assignment
P=Path('/mnt/data/qwen_next')
s=(P/'qwen_joint_graph_halt.py').read_text().split('net=Joint();')[0]
exec(compile(s,str(P/'qwen_joint_graph_halt.py'),'exec'))
room_names=['bathroom','bedroom','garden','hallway','kitchen','office']
paths=[('qwen_seed61425','qwen_joint_graph_halt_curriculum/joint_graph_halt_curriculum_seed61425.pt'),('qwen_seed61426','qwen_joint_graph_halt_curriculum/joint_graph_halt_curriculum_seed61426.pt'),('permuted_seed61425','qwen_joint_graph_halt_permuted/joint_graph_halt_permuted_seed61425.pt')]
results={}
for label,fn in paths:
 net=Joint();net.load_state_dict(torch.load(P/fn,weights_only=True));net.eval()
 with torch.no_grad():
  h=net.e(net.feat);A=F.softmax(net.a(h),-1);B=F.softmax(net.b(h),-1)
  rel=torch.zeros(6,8);cts=torch.zeros(6)
  events=0
  for i,sent in enumerate(index):
   m=pat.fullmatch(sent)
   if not m:continue
   x,d,y=m.groups();
   for room,dist in [(x,A[i]),(y,B[i])]:rel[room_names.index(room)]+=dist;cts[room_names.index(room)]+=1
   events+=1
  rel=rel/cts[:,None];R,C=linear_sum_assignment(-rel.numpy())
  correspondence={room_names[int(i)]:int(j) for i,j in zip(R,C)}
  pur=float(rel[R,C].mean());mass=float(rel.max(-1).values.mean())
  event_correct=[]
  for i,sent in enumerate(index):
   m=pat.fullmatch(sent)
   if not m:continue
   x,d,y=m.groups();event_correct.append([int(A[i].argmax())==correspondence[x],int(B[i].argmax())==correspondence[y]])
  score=sum(sum(e) for e in event_correct);n=2*len(event_correct)
  results[label]={'room_to_anonymous_node':correspondence,'mean_assigned_room_node_probability':round(pur,4),'mean_room_argmax_mass':round(mass,4),'event_endpoints_correct':score,'event_endpoints_total':n,'room_node_distributions':{room_names[i]:[round(float(z),3) for z in rel[i]] for i in range(6)},'notes':'posthoc Hungarian alignment to room labels only for measurement; no room-to-node labels used in training.'}
  print(label,'correct endpoint assignments',score,'/',n,'mean_probs',pur,flush=True)
(P/'qwen_joint_graph_halt_curriculum'/'node_binding_audit.json').write_text(json.dumps(results,indent=2))
