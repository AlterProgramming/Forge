import torch,json,time,argparse,platform
from torch import nn
import torch.nn.functional as F
p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=400);p.add_argument('--seeds',type=int,default=3);a=p.parse_args()
torch.set_num_threads(2)
class Net(nn.Module):
 def __init__(self):
  super().__init__();self.enc=nn.Sequential(nn.Linear(10,64),nn.Tanh(),nn.Linear(64,32));self.rewrite=nn.Sequential(nn.Linear(64,64),nn.Tanh(),nn.Linear(64,32));self.read=nn.Linear(32,4)
 def forward(self,x,mode,ablate='none'):
  b,t,_=x.shape;s=x.new_zeros(b,32);outs=[]
  for j in range(t):
   e=self.enc(x[:,j]);prev=s if mode!='blind' and ablate!='blind' else torch.zeros_like(s)
   delta=self.rewrite(torch.cat([prev,e],-1))
   s=delta if mode=='replace' else s+delta
   if ablate=='shuffle' and j==t//2:s=s.roll(1,0)
   if ablate=='zero' and j==t//2:s=torch.zeros_like(s)
   outs.append(self.read(s))
  return torch.stack(outs,1)
def sample(n,t,task,seed):
 g=torch.Generator().manual_seed(seed);keys=torch.randint(0,4,(n,t),generator=g);values=torch.randint(0,2,(n,t),generator=g);states=torch.zeros(n,4,dtype=torch.long);xs=[];ys=[]
 for j in range(t):
  k=keys[:,j];v=values[:,j];toggle=(torch.randint(0,2,(n,),generator=g) if task=='mixed' else torch.full((n,),int(task=='toggle')))
  xs.append(torch.cat([F.one_hot(k,4).float(),F.one_hot(v,2).float(),F.one_hot(toggle,2).float(),torch.ones(n,2)],-1));idx=torch.arange(n)
  old=states[idx,k].clone();states[idx,k]=torch.where(toggle.bool(),old^v,v)
  ys.append(states.clone())
 return torch.stack(xs,1),torch.stack(ys,1).float()
start=time.time();results=[]
for mode in ['dependent','blind','replace']:
 for seed in range(a.seeds):
  torch.manual_seed(seed+2026);m=Net();opt=torch.optim.AdamW(m.parameters(),lr=.003);x,y=sample(256,6,'mixed',seed+100)
  for i in range(a.steps):
   opt.zero_grad();loss=F.binary_cross_entropy_with_logits(m(x,mode),y);loss.backward();nn.utils.clip_grad_norm_(m.parameters(),1);opt.step()
  row={'mode':mode,'seed':seed,'parameters':sum(z.numel() for z in m.parameters()),'train_steps':a.steps,'final_train_loss':round(loss.item(),6)}
  with torch.no_grad():
   for task in ['mixed','overwrite','toggle']:
    for length in [6,24,48]:
     vx,vy=sample(512,length,task,10000+seed*100+length+{'mixed':0,'overwrite':1,'toggle':2}[task])
     for ab in ['none','blind','shuffle','zero']:
      pred=m(vx,mode,ab);row[f'{task}_L{length}_{ab}']=round(((pred>0)==vy.bool()).float().mean().item(),6)
  results.append(row);print(json.dumps(row),flush=True)
out={'config':vars(a),'torch':torch.__version__,'platform':platform.platform(),'cpu_threads':2,'elapsed_seconds':time.time()-start,'results':results,'limitations':'No Qwen weights or Qwen readback; small fixed-key synthetic tasks; controls share architecture and parameters but additive versus replacement arithmetic differs; evaluation uses new seeded sequences, not independent semantic tasks; no general capability established.'}
with open('rewrite_mixed_results.json','w') as f:json.dump(out,f,indent=2)
