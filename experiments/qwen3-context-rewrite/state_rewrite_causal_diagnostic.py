import torch,random,json,time,argparse
from torch import nn
import torch.nn.functional as F
p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=500);p.add_argument('--seeds',type=int,default=3);a=p.parse_args()
torch.set_num_threads(2)
# Two independent sequence transformations: keyed overwrite and keyed toggle.
# Input event is (key,value,operation), output is queried binary value for all keys.
# The learned rewrite model uses s_{t+1}=s_t+f(s_t,event_t), and reads updated state at every step.
# A matched nonrewrite model uses an event-only representation, with the same parameters.
class Net(nn.Module):
 def __init__(self):
  super().__init__();self.enc=nn.Sequential(nn.Linear(10,64),nn.Tanh(),nn.Linear(64,32));self.rewrite=nn.Sequential(nn.Linear(64,64),nn.Tanh(),nn.Linear(64,32));self.read=nn.Linear(32,4)
 def forward(self,x,mode='rewrite',shuffle=False):
  b,t,_=x.shape;s=torch.zeros(b,32);outs=[]
  for j in range(t):
   e=self.enc(x[:,j]);inp=torch.cat([s if mode in ('rewrite','recurrent_replace') else torch.zeros_like(s),e],-1)
   delta=self.rewrite(inp)
   if mode=='rewrite':s=s+delta
   elif mode=='recurrent_replace':s=torch.tanh(delta)
   elif mode=='event_only':s=delta # same trainable weights, no retained state
   if shuffle:s=s.roll(1,0)
   outs.append(self.read(s))
  return torch.stack(outs,1)
def sample(n,t,task,seed):
 g=torch.Generator().manual_seed(seed);keys=torch.randint(0,4,(n,t),generator=g);values=torch.randint(0,2,(n,t),generator=g);states=torch.zeros(n,4,dtype=torch.long);xs=[];ys=[]
 for j in range(t):
  k=keys[:,j];v=values[:,j];x=F.one_hot(k,4).float();ev=F.one_hot(v,2).float();op=torch.tensor([1.,0.] if task=='overwrite' else [0.,1.]).repeat(n,1)
  xs.append(torch.cat([x,ev,op,torch.ones(n,2)],-1));idx=torch.arange(n)
  if task=='overwrite':states[idx,k]=v
  else:states[idx,k]^=v
  ys.append(states.clone())
 return torch.stack(xs,1),torch.stack(ys,1).float()
results=[];start=time.time()
for task in ['overwrite','toggle']:
 for mode in ['rewrite','recurrent_replace','event_only']:
  for seed in range(a.seeds):
   torch.manual_seed(seed+20);m=Net();opt=torch.optim.AdamW(m.parameters(),lr=.003);x,y=sample(256,6,task,seed+100)
   for i in range(a.steps):
    opt.zero_grad();out=m(x,mode);loss=F.binary_cross_entropy_with_logits(out,y);loss.backward();nn.utils.clip_grad_norm_(m.parameters(),1);opt.step()
   row={'task':task,'mode':mode,'seed':seed,'parameters':sum(z.numel() for z in m.parameters()),'train_steps':a.steps}
   for length in [6,12,24]:
    vx,vy=sample(512,length,task,seed+1000+length)
    with torch.no_grad():
     pred=m(vx,mode);row[f'length_{length}_bit_acc']=round(((pred>0)==vy.bool()).float().mean().item(),5)
     row[f'length_{length}_all_bits_acc']=round(((pred>0)==vy.bool()).all(-1).float().mean().item(),5)
     if mode in ('rewrite','recurrent_replace'):
      ab=m(vx,mode,shuffle=True);row[f'length_{length}_shuffled_state_bit_acc']=round(((ab>0)==vy.bool()).float().mean().item(),5)
   results.append(row);print(json.dumps(row),flush=True)
print('elapsed_sec',round(time.time()-start,2))
with open('results.json','w') as f:json.dump({'config':vars(a),'results':results,'elapsed_sec':round(time.time()-start,2),'disclosure':'Synthetic CPU interface diagnostic; Qwen weights not loaded; no evidence of Qwen self-rewriting.'},f,indent=2)
