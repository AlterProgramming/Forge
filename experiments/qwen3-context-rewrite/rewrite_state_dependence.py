import torch, random, json, time, argparse
from torch import nn
import torch.nn.functional as F
p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=350);p.add_argument('--seeds',type=int,default=2);a=p.parse_args()
torch.set_num_threads(2)
class Net(nn.Module):
 def __init__(self):
  super().__init__();self.enc=nn.Sequential(nn.Linear(10,64),nn.Tanh(),nn.Linear(64,32));self.rewrite=nn.Sequential(nn.Linear(64,64),nn.Tanh(),nn.Linear(64,32));self.read=nn.Linear(32,4)
 def forward(self,x,mode,ablate='none'):
  b,t,_=x.shape;s=x.new_zeros(b,32);outs=[]
  for j in range(t):
   e=self.enc(x[:,j]);state_input=s if mode=='dependent' and ablate!='blind' else torch.zeros_like(s)
   delta=self.rewrite(torch.cat([state_input,e],-1))
   if ablate=='no_update' and j>=3:delta=torch.zeros_like(delta)
   s=s+delta
   if ablate=='shuffle' and j==t//2:s=s.roll(1,0)
   outs.append(self.read(s))
  return torch.stack(outs,1)
def sample(n,t,task,seed):
 g=torch.Generator().manual_seed(seed);keys=torch.randint(0,4,(n,t),generator=g);values=torch.randint(0,2,(n,t),generator=g);states=torch.zeros(n,4,dtype=torch.long);xs=[];ys=[]
 for j in range(t):
  k=keys[:,j];v=values[:,j];op=torch.tensor([1.,0.] if task=='overwrite' else [0.,1.]).repeat(n,1)
  xs.append(torch.cat([F.one_hot(k,4).float(),F.one_hot(v,2).float(),op,torch.ones(n,2)],-1));idx=torch.arange(n)
  if task=='overwrite':states[idx,k]=v
  else:states[idx,k]^=v
  ys.append(states.clone())
 return torch.stack(xs,1),torch.stack(ys,1).float()
results=[];start=time.time()
for task in ['overwrite','toggle']:
 for mode in ['dependent','blind']:
  for seed in range(a.seeds):
   torch.manual_seed(seed+20);m=Net();opt=torch.optim.AdamW(m.parameters(),lr=.003);x,y=sample(256,6,task,seed+100)
   for i in range(a.steps):
    opt.zero_grad();loss=F.binary_cross_entropy_with_logits(m(x,mode),y);loss.backward();nn.utils.clip_grad_norm_(m.parameters(),1);opt.step()
   row={'task':task,'mode':mode,'seed':seed,'parameters':sum(z.numel() for z in m.parameters()),'steps':a.steps}
   for length in [6,12,24]:
    vx,vy=sample(512,length,task,seed+1000+length)
    with torch.no_grad():
     for ab in ['none','blind','shuffle','no_update']:
      pred=m(vx,mode,ab);row[f'L{length}_{ab}_bit_acc']=round(((pred>0)==vy.bool()).float().mean().item(),5)
   results.append(row);print(json.dumps(row),flush=True)
print('elapsed',time.time()-start)
with open('rewrite_state_dependence_results.json','w') as f:json.dump({'config':vars(a),'results':results,'elapsed_sec':time.time()-start,'limitations':'Synthetic tasks; Qwen not loaded; no frozen Qwen readback; only two seeds; no robust generalization claim'},f,indent=2)
