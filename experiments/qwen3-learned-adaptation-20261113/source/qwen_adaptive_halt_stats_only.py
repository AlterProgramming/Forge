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
  if self.whole == 'stats_only':
   f=torch.cat([torch.zeros_like(f[:,:,:-5]),f[:,:,-5:]],-1)
  elif not self.whole:
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
 torch.manual_seed(args.seed+800+(2 if whole=='stats_only' else int(whole)));net=SharedStop(whole);opt=torch.optim.AdamW(net.parameters(),lr=.002,weight_decay=.001)
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
 torch.save(net.state_dict(),O/('stop_stats_only.pt' if whole=='stats_only' else 'stop_graph.pt' if whole else 'stop_no_graph.pt'))
 return {'validation':val,'test_in_distribution':testscore,'test_unseen_fourth_step':ood,'best_epoch':epoch,'curve':curve},(p==teY).int().tolist(),(p4==t4Y).int().tolist()

stats,pp,po=train_model('stats_only')
(O/'stats_only_results.json').write_text(json.dumps({'stats_only':stats,'test_id_correct_vector':pp,'test_ood_correct_vector':po,'seed':args.seed,'feature_checkpoint':args.checkpoint},indent=2))
