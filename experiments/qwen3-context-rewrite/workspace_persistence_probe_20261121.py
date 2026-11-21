#!/usr/bin/env python3
"""CPU-only workspace persistence diagnostic. NOT a Qwen experiment."""
import torch,random,json,time,argparse
from torch import nn
from torch.nn import functional as F
p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=220);p.add_argument('--output',default='results.json');a=p.parse_args()
torch.set_num_threads(2); start=time.time()
class Net(nn.Module):
 def __init__(self):
  super().__init__();self.event=nn.Embedding(8,16);self.up=nn.Sequential(nn.Linear(32,32),nn.Tanh(),nn.Linear(32,16));self.read=nn.Sequential(nn.Linear(16,16),nn.Tanh(),nn.Linear(16,16));self.head=nn.Linear(16,4)
 def forward(self,x,mode='rewrite',intervention='none'):
  s=torch.zeros(x.shape[0],16)
  for t in range(x.shape[1]):
   e=self.event(x[:,t]);prev=s if mode=='rewrite' else torch.zeros_like(s)
   delta=self.up(torch.cat([e,prev],-1))
   s=s+delta if mode=='rewrite' else delta
   r=self.read(s)
   if intervention=='shuffle':s=s.roll(1,0)
   elif intervention=='zero':s=torch.zeros_like(s)
  return self.head(r)
def data(n,length,seed):
 rng=random.Random(seed);xx=[];yy=[]
 for i in range(n):
  state=[0,0];seq=[]
  for j in range(length):
   op=rng.randrange(8);k=op//4;kind=op%4
   if kind==0:state[k]=0
   if kind==1:state[k]=1
   if kind==2:state[k]^=1
   seq.append(op)
  xx.append(seq);yy.append(state[0]*2+state[1])
 return torch.tensor(xx),torch.tensor(yy)
results=[]
for seed in (13,29,47):
 for mode in ('rewrite','blind'):
  torch.manual_seed(seed);m=Net();opt=torch.optim.AdamW(m.parameters(),lr=.002)
  tr=data(512,6,1000+seed)
  for step in range(a.steps):
   idx=torch.randint(512,(64,));log=m(tr[0][idx],mode);loss=F.cross_entropy(log,tr[1][idx]);opt.zero_grad();loss.backward();opt.step()
  row={'seed':seed,'mode':mode,'parameters':sum(v.numel() for v in m.parameters())}
  with torch.no_grad():
   for length in (6,24,48):
    x,y=data(512,length,20000+length)
    for intervention in (('none','shuffle','zero') if length==24 else ('none',)):
     row[f'L{length}_{intervention}']=round((m(x,mode,intervention).argmax(-1)==y).float().mean().item(),6)
  results.append(row);print(row,flush=True)
payload={'config':vars(a),'results':results,'seconds':time.time()-start,'limitations':'No Qwen loaded; synthetic diagnostic, no matched measured FLOPs. Blind computes same update network but zeros prior state and replaces workspace; shuffle/zero change state distribution. Final-only labels, no intermediate supervision. Fixed training seeds 13/29/47, training data seeds 1013/1029/1047; test seeds 20006/20024/20048. No claim of general capability.'}
open(a.output,'w').write(json.dumps(payload,indent=2));print('seconds',payload['seconds'])
