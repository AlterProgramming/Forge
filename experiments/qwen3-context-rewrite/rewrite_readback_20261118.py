import argparse,json,time,platform
import torch
from torch import nn
import torch.nn.functional as F
p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=300);p.add_argument('--seeds',type=int,default=3);p.add_argument('--threads',type=int,default=2);a=p.parse_args()
torch.set_num_threads(a.threads)
class Net(nn.Module):
 def __init__(self):
  super().__init__()
  self.enc=nn.Sequential(nn.Linear(10,64),nn.Tanh(),nn.Linear(64,32))
  self.readback=nn.Sequential(nn.Linear(32,32),nn.Tanh())
  self.update=nn.Sequential(nn.Linear(64,64),nn.Tanh(),nn.Linear(64,32))
  self.out=nn.Linear(32,4)
 def forward(self,x,mode,ablate='none'):
  s=x.new_zeros(x.shape[0],32);ys=[]
  for t in range(x.shape[1]):
   e=self.enc(x[:,t]);r=self.readback(s)
   if mode=='blind' or ablate=='blind':r=torch.zeros_like(r)
   if ablate=='shuffle':r=r.roll(1,0)
   if ablate=='lag' and t>0:r=old_r
   d=self.update(torch.cat((e,r),-1))
   old_r=self.readback(s)
   if mode=='replace':s=d
   else:s=s+d
   ys.append(self.out(s))
  return torch.stack(ys,1)
def sample(n,T,task,seed):
 g=torch.Generator().manual_seed(seed)
 k=torch.randint(0,4,(n,T),generator=g);v=torch.randint(0,2,(n,T),generator=g)
 st=torch.zeros(n,4,dtype=torch.long);xs=[];ys=[]
 for t in range(T):
  op=torch.randint(0,2,(n,),generator=g) if task=='mixed' else torch.full((n,),int(task=='toggle'))
  xs.append(torch.cat((F.one_hot(k[:,t],4),F.one_hot(v[:,t],2),F.one_hot(op,2),torch.ones(n,2)),1).float())
  idx=torch.arange(n);old=st[idx,k[:,t]].clone();st[idx,k[:,t]]=torch.where(op.bool(),old^v[:,t],v[:,t]);ys.append(st.clone())
 return torch.stack(xs,1),torch.stack(ys,1).float()
t0=time.perf_counter();rows=[]
for mode in ('rewrite','blind','replace'):
 for seed in range(a.seeds):
  torch.manual_seed(20261118+seed);m=Net();opt=torch.optim.AdamW(m.parameters(),lr=.003)
  x,y=sample(256,6,'mixed',seed+101)
  for step in range(a.steps):
   opt.zero_grad();loss=F.binary_cross_entropy_with_logits(m(x,mode),y);loss.backward();nn.utils.clip_grad_norm_(m.parameters(),1);opt.step()
  row={'mode':mode,'seed':seed,'parameters':sum(z.numel() for z in m.parameters()),'loss':float(loss.item())}
  with torch.no_grad():
   for task in ('mixed','overwrite','toggle'):
    for T in (6,24,48):
     vx,vy=sample(512,T,task,900000+seed*1000+T+{'mixed':0,'overwrite':100,'toggle':200}[task])
     for ab in ('none','blind','shuffle','lag'):
      z=m(vx,mode,ab);row[f'{task}_L{T}_{ab}']=float(((z>0)==vy.bool()).float().mean().item())
  rows.append(row);print(json.dumps(row),flush=True)
out={'experiment':'learned readback into state-dependent additive update vs blind and replacement controls','args':vars(a),'torch':torch.__version__,'platform':platform.platform(),'elapsed_s':time.perf_counter()-t0,'rows':rows,'limitations':['No Qwen checkpoint loaded; learned readback is a surrogate network, not Qwen reading rewritten state','Synthetic 4-key tasks; new deterministic sequences are not semantic OOD','Equal parameter count, optimization steps, examples, and state size; not equal executed FLOPs because blind still computes readback but zeroes it','Repeated operations; causal readback interventions at inference; no causal proof of semantic state use','Training uses fixed 256 sequences, not resampling each step']}
with open('rewrite_readback_results_20261118.json','w') as f:json.dump(out,f,indent=2)
