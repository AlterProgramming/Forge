#!/usr/bin/env python3
import torch,random,time,json,argparse
from torch import nn
from torch.nn import functional as F
p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=300);p.add_argument('--output',default='results.json');a=p.parse_args()
torch.set_num_threads(2);start=time.time()
class Net(nn.Module):
 def __init__(self):
  super().__init__();self.event=nn.Embedding(8,16);self.update=nn.Sequential(nn.Linear(48,32),nn.Tanh(),nn.Linear(32,16));self.head=nn.Linear(16,4)
 def forward(self,x,mode,ablate='none'):
  s=torch.zeros(x.size(0),16)
  for t in range(x.size(1)):
   r=torch.tanh(s*1.7)
   if ablate=='shuffle':r=r.roll(1,0)
   if ablate=='zero':r=torch.zeros_like(r)
   e=self.event(x[:,t]);delta=self.update(torch.cat([e,r,s if mode=='rewrite' else torch.zeros_like(s)],-1))
   s=s+delta if mode=='rewrite' else delta
  return self.head(torch.tanh(s*1.7))
def data(n,L,seed):
 rng=random.Random(seed);X=[];Y=[]
 for _ in range(n):
  st=[0,0];seq=[]
  for t in range(L):
   v=rng.randrange(8);k=v//4;op=v%4
   if op==0:st[k]=0
   elif op==1:st[k]=1
   elif op==2:st[k]^=1
   seq.append(v)
  X.append(seq);Y.append(st[0]*2+st[1])
 return torch.tensor(X),torch.tensor(Y)
rows=[]
for seed in [13,29,47]:
 tr=data(512,6,1000+seed)
 for mode in ['rewrite','replacement']:
  torch.manual_seed(seed);net=Net();opt=torch.optim.AdamW(net.parameters(),lr=.002)
  for i in range(a.steps):
   idx=torch.randint(512,(64,));loss=F.cross_entropy(net(tr[0][idx],mode),tr[1][idx]);opt.zero_grad();loss.backward();opt.step()
  row={'seed':seed,'mode':mode,'parameters':sum(p.numel() for p in net.parameters())}
  with torch.no_grad():
   for L in [6,24,48]:
    x,y=data(512,L,20000+L)
    for ab in (['none','shuffle','zero'] if L==24 else ['none']):
     row[f'L{L}_{ab}']=round((net(x,mode,ab).argmax(-1)==y).float().mean().item(),6)
  rows.append(row);print(row,flush=True)
payload={'results':rows,'seconds':time.time()-start,'config':vars(a),'data':'512 train examples per seed length6, heldout 512 examples length 6/24/48; seeds train 1000+seed, test 20000+length','limitations':'No Qwen checkpoint loaded. Fixed tanh readback proxy is not Qwen. Equal parameters and event/update calls, not matched FLOPs due to residual and different dependencies. Synthetic keyed operations, not general capability. Final outcome supervision only.'}
open(a.output,'w').write(json.dumps(payload,indent=2));print('seconds',payload['seconds'])
