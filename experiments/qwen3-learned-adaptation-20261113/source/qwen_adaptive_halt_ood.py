#!/usr/bin/env python3
"""Train a shared read/rewrite/STOP rule on 1-3 hops, test on unseen 4-hop relations.
Frozen Qwen3-derived graph; no step-specific STOP labels, only final route answers.
All graph traversal operators share exactly the same weights at every depth.
"""
import time, pathlib, copy, math, json, collections
import torch
from torch import nn
import torch.nn.functional as F
import random
P=pathlib.Path('/mnt/data/qwen_next')
source=(P/'qwen_operator_autoselect.py').read_text().split('class Router(nn.Module):')[0]
exec(compile(source,str(P/'qwen_operator_autoselect.py'),'exec'))
O=P/'qwen_adaptive_halt_ood';O.mkdir(exist_ok=True)

def enrich4(row,where):
 g=exact_graph(row['hist']);options=[]
 for a in sorted(g):
  for b in sorted(g):
   if a==b:continue
   r=path(g,a,b)
   if len(r)==4:options.append((a,b,r))
 if not options:return []
 rng=random.Random(args.seed*100003+row['sid']+where*100013)
 a,b,r=rng.choice(options);q=f'How do you go from the {a} to the {b}?'
 return [(row['hist'],q,r,4,row['sid'])]

test4=[it for row in test_rows for it in enrich4(row,1)]
valid4=[it for row in train_rows if row['sid'] in va for it in enrich4(row,0)]
print('OOD held-out hop4 questions',len(test4),len(valid4),flush=True)

@torch.no_grad()
def features(rows,batch=512):
 stop_feats=[];path_probs=[];label=[];true_depth=[];relative=[]
 for j in range(0,len(rows),batch):
  chunk=rows[j:j+batch];B=len(chunk)
  qq=torch.tensor([idx[r[1]] for r in chunk]);ev=torch.tensor([[idx[s] for s in r[0]] for r in chunk]);g=graph_event[ev].sum(1)
  s=src[qq];t=tgt[qq]; state=s[:,None,:];prev_sum=torch.zeros(B)
  feats=[]; probs=[]
  for k in (1,2,3,4):
   state=torch.einsum('bpi,bdij->bpdj',state,g).reshape(B,-1,N)
   w=torch.einsum('bpn,bn->bp',state,t).clamp_min(1e-12)
   lp=F.log_softmax(w.log(),dim=-1)
   probs.append(lp)
   norm=F.softmax(w.log(),dim=-1)
   ent=-(norm*norm.clamp_min(1e-12).log()).sum(-1)/math.log(4**k)
   total=w.sum(-1);mx=w.max(-1).values
   # Identical function at every step; no step-index embedding.
   m=torch.stack([total.log1p(),mx.log1p(),ent,(total-prev_sum).tanh(),mx/(total+1e-7)],-1)
   feats.append(torch.cat([Q[qq],m],-1))
   prev_sum=total
  yf=[]
  for r in chunk:
   route=r[2];k=len(route);code=0
   for d in route:code=code*4+D.index(d)
   offset={1:0,2:4,3:20,4:84}[k];yf.append(offset+code)
  stop_feats.append(torch.stack(feats,1));path_probs.append(torch.cat(probs,1));label.extend(yf);true_depth.extend(r[3] for r in chunk)
 return torch.cat(stop_feats),torch.cat(path_probs),torch.tensor(label),torch.tensor(true_depth)
trF,trP,trY,trD=features(train)
vaF,vaP,vaY,vaD=features(valid)
teF,teP,teY,teD=features(test)
t4F,t4P,t4Y,t4D=features(test4)
print('PRECOMPUTED',time.time()-st,'features',trF.shape,'4-hop score',int((t4P[:,84:].argmax(-1)+84==t4Y).sum()),'/',len(test4),flush=True)

class SharedStop(nn.Module):
 def __init__(self,whole=True):
  super().__init__();self.whole=whole
  self.net=nn.Sequential(nn.LayerNorm(trF.shape[-1]),nn.Linear(trF.shape[-1],64),nn.GELU(),nn.Linear(64,1))
 def forward(self,f):
  if not self.whole:
   f=torch.cat([f[:,:,:-5],torch.zeros_like(f[:,:,-5:])],-1)
  a=self.net(f).squeeze(-1);haz=torch.sigmoid(a[:,:3]).clamp(1e-6,1-1e-6)
  p=[haz[:,0],(1-haz[:,0])*haz[:,1],(1-haz[:,0])*(1-haz[:,1])*haz[:,2],
     (1-haz[:,0])*(1-haz[:,1])*(1-haz[:,2])]
  return torch.stack(p,1)

def mix(net,f,probs):
 halt=net(f).clamp_min(1e-9).log()
 return torch.cat([probs[:,:4]+halt[:,0,None],probs[:,4:20]+halt[:,1,None],probs[:,20:84]+halt[:,2,None],probs[:,84:]+halt[:,3,None]],1)

def score(net,f,probs,y,depth):
 net.eval()
 with torch.no_grad():
  out=mix(net,f,probs);pr=out.argmax(-1);length=torch.bucketize(pr,torch.tensor([4,20,84]))+1
  return {'correct':int((pr==y).sum()),'n':len(y),'accuracy':round((pr==y).float().mean().item(),5),
          'predicted_depth_accuracy':round((length==depth).float().mean().item(),5),
          'predicted_depth_counts':{str(k):int((length==k).sum()) for k in (1,2,3,4)}},pr

def train_model(whole):
 torch.manual_seed(args.seed+800+int(whole));net=SharedStop(whole);opt=torch.optim.AdamW(net.parameters(),lr=.002,weight_decay=.001)
 best=-1;epoch=0;state=None;curve=[]
 for ep in range(1,81):
  net.train();perm=torch.randperm(len(trY))
  for ii in perm.split(256):
   z=mix(net,trF[ii],trP[ii]);loss=F.nll_loss(z,trY[ii]);opt.zero_grad(set_to_none=True);loss.backward();opt.step()
  if ep==1 or ep%5==0:
   v,_=score(net,vaF,vaP,vaY,vaD)
   if v['correct']>best:best=v['correct'];state=copy.deepcopy(net.state_dict());epoch=ep
   curve.append([ep,v['correct']]);
   if ep%15==0 or ep==1:print('TRAIN',whole,'ep',ep,'valid',v['correct'],'best',best,flush=True)
 net.load_state_dict(state)
 val,_=score(net,vaF,vaP,vaY,vaD);testscore,p=score(net,teF,teP,teY,teD);ood,p4=score(net,t4F,t4P,t4Y,t4D)
 print('FINAL',whole,'train_depths_1to3',testscore,'novel_depth4',ood,'checkpoint_epoch',epoch,flush=True)
 torch.save(net.state_dict(),O/('stop_graph.pt' if whole else 'stop_no_graph.pt'))
 return {'validation':val,'test_in_distribution':testscore,'test_unseen_fourth_step':ood,'best_epoch':epoch,'curve':curve},(p==teY).int().tolist(),(p4==t4Y).int().tolist()

full,p,oodvec=train_model(True)
blind,pb,oddb=train_model(False)
# Reference (not a learned policy): choose the horizon with highest target reachability confidence.
def confidence_baseline(f,probs,y):
 feats=f[:,:,-5:];z=feats[:,:,0] # log1p total route mass at each depth
 lenchoice=z.argmax(-1)
 out=[]
 for i,k in enumerate(lenchoice.tolist()):
  a,b=[(0,4),(4,20),(20,84),(84,340)][k];out.append(int(probs[i,a:b].argmax())+a)
 return {'correct':int((torch.tensor(out)==y).sum()),'n':len(y),'selected':{str(k):int((lenchoice==k-1).sum()) for k in (1,2,3,4)}}
base={'trained_depths':confidence_baseline(teF,teP,teY),'unseen_depth4':confidence_baseline(t4F,t4P,t4Y)}
print('READOUT_REFERENCE',base,flush=True)
result={'hypothesis':'shared learned continue/STOP computation generalizes to unseen longer paths','training_depths':[1,2,3],'held_out_depth':4,'pretrained_qwen3_weights_updated':False,'graph_weights_updated':False,'trainable_components':['small shared STOP gate using Qwen question and latent reachability features'],'predefined_mechanisms':['relational graph memory','parameter-free one-hop traversal reused at arbitrary depth','latent predicate target reachability'],'answer_supervision_only':True,'important_caveat':'answer sequences reveal path length implicitly; no explicit step labels. Graph model already trained on official bAbI task19 two-hop. Mixed paths are deterministic extra queries derived from the original stories, not an independently authored benchmark. Same known room/sentence/question vocabulary at train and test. The experiment tests operation reuse and learned termination, not invention of a new primitive.','example_counts':{'train_1_3':len(train),'validation_1_3':len(valid),'test_1_3':len(test),'test_4':len(test4)},'freeze_checkpoint':args.checkpoint,'learned_gate':full,'graph_blind_gate':blind,'nonlearned_confidence_baseline':base,'test_id_correct_vector':p,'test_ood_correct_vector':oodvec,'test_id_blind_vector':pb,'test_ood_blind_vector':oddb,'runtime_seconds':round(time.time()-st,2)}
(O/'results.json').write_text(json.dumps(result,indent=2))
print('SAVED',O/'results.json','SECONDS',round(time.time()-st,2),flush=True)
