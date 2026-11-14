import sys,torch,json
from pathlib import Path
P=Path('/mnt/data/qwen_next')
source=(P/'qwen_joint_graph_halt.py').read_text().split('net=Joint();')[0]
exec(compile(source,str(P/'qwen_joint_graph_halt.py'),'exec'))
def predictions(net,data):
 net.eval();z=[]
 with torch.no_grad():
  for c in torch.arange(len(data[2])).split(256):
   a,b,y,d=[v[c] for v in data];s,*_=net(a,b);z.extend(s.argmax(-1).tolist())
 return z
out={}
pathmap=[
 ('no_curriculum_seed61425','qwen_joint_graph_halt','joint_graph_halt_seed61425.pt'),
 ('curriculum_seed61425','qwen_joint_graph_halt_curriculum','joint_graph_halt_curriculum_seed61425.pt'),
 ('curriculum_seed61426','qwen_joint_graph_halt_curriculum','joint_graph_halt_curriculum_seed61426.pt'),
 ('permuted_features_seed61425','qwen_joint_graph_halt_permuted','joint_graph_halt_permuted_seed61425.pt'),
 ('permuted_features_seed61426','qwen_joint_graph_halt_permuted','joint_graph_halt_permuted_seed61426.pt'),
 ('curriculum_seed61427','qwen_joint_graph_halt_curriculum','joint_graph_halt_curriculum_seed61427.pt')]
for name,directory,file in pathmap:
 net=Joint();net.load_state_dict(torch.load(P/directory/file,weights_only=True))
 p=predictions(net,te);o=predictions(net,t4)
 out[name]={'test_pred':p,'ood_pred':o,'correct_test':[int(a==b) for a,b in zip(p,te[2].tolist())],'correct_ood':[int(a==b) for a,b in zip(o,t4[2].tolist())]}
 print(name,sum(out[name]['correct_test']),sum(out[name]['correct_ood']),flush=True)
O=P/'qwen_joint_graph_halt_curriculum'
(O/f'paired_test_predictions_eval_seed{args.seed}.json').write_text(json.dumps({'evaluation_seed':args.seed,'test_ground_truth':te[2].tolist(),'ood_ground_truth':t4[2].tolist(),'test_story_ids':[r[4] for r in test],'ood_story_ids':[r[4] for r in test4],'methods':out}))
