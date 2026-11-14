#!/usr/bin/env python3
"""Joint outcome-only training: Qwen3 latent feature -> learned graph -> shared traversal -> learned halt.
Task: mixed 1/2/3 hop questions from bAbI Task19 graph stories, unseen hop4 test.
The relational graph/tensor-traversal *architectural prior* remains user-supplied.
No checkpoint for the graph or router is loaded: both start randomly.
"""
import torch,math,time,copy,random,json,sys
from torch import nn
import torch.nn.functional as F
from pathlib import Path
P=Path('/mnt/data/qwen_next');O=P/'qwen_joint_graph_halt';O.mkdir(exist_ok=True)
# Import only dataset+exact-label construction code, not the frozen controller, training loop.
code=(P/'qwen_operator_autoselect.py').read_text().split('def fixed_projection(')[0]
# This defines seed, parse, enrich, feat, and dataset helpers.
exec(compile(code,str(P/'qwen_operator_autoselect.py'),'exec'))
# Use Qwen feature centering / unsupervised whitening; no answer labels used for geometry.
with torch.no_grad():
 center=feat-feat.mean(0,keepdim=True)
 u,s,v=torch.linalg.svd(center,full_matrices=False)
 rank=int((s>s[0]*1e-5).sum())
 feat=(u[:,:rank]@v[:rank])*8.0

# After geometry, regenerate disjoint mixed-horizon datasets.
train_rows=parse(P/'qa19_10k_train.txt');test_rows=parse(P/'qa19_10k_test.txt')
ids=list(range(10000));random.Random(90117).shuffle(ids);va=set(ids[:1000])
train=[it for r in train_rows if r['sid'] not in va for it in enrich(r,0)]
valid=[it for r in train_rows if r['sid'] in va for it in enrich(r,0)]
test=[it for r in test_rows for it in enrich(r,1)]
def augment4(row,flag):
 g=exact_graph(row['hist']);cand=[]
 for a in sorted(g):
  for b in sorted(g):
   if a!=b:
    steps=path(g,a,b)
    if len(steps)==4:cand.append((a,b,steps))
 if not cand:return []
 rng=random.Random(args.seed*100003+row['sid']+flag*100013)
 a,b,steps=rng.choice(cand)
 return [(row['hist'],f'How do you go from the {a} to the {b}?',steps,4,row['sid'])]
test4=[it for r in test_rows for it in augment4(r,1)]

def prepare(rows):
 X=torch.tensor([[idx[s] for s in r[0]] for r in rows]);Q=torch.tensor([idx[r[1]] for r in rows]);Y=[];D=[]
 for r in rows:
  code=0
  for s in r[2]:code=code*4+DIRECTIONS.index(s)
  Y.append({1:0,2:4,3:20,4:84}[len(r[2])]+code)
  D.append(len(r[2]))
 return X,Q,torch.tensor(Y),torch.tensor(D)
DIRECTIONS=['n','s','e','w']
tr=prepare(train);v=prepare(valid);te=prepare(test);t4=prepare(test4)
print('DATA',len(train),len(valid),len(test),len(test4),'features',feat.shape,'seconds',round(time.time()-st,1),flush=True)

class Joint(nn.Module):
 def __init__(self):
  super().__init__();self.n=8;self.register_buffer('feat',feat)
  self.e=nn.Sequential(nn.LayerNorm(1024),nn.Linear(1024,128),nn.GELU(),nn.Linear(128,96),nn.GELU())
  self.q=nn.Sequential(nn.LayerNorm(1024),nn.Linear(1024,128),nn.GELU(),nn.Linear(128,96),nn.GELU())
  self.a=nn.Linear(96,8);self.b=nn.Linear(96,8);self.dir=nn.Linear(96,4)
  self.src=nn.Linear(96,8);self.tgt=nn.Linear(96,8)
  self.stop=nn.Sequential(nn.LayerNorm(101),nn.Linear(101,64),nn.GELU(),nn.Linear(64,1))
  self.register_buffer('op',torch.tensor([1,0,3,2]))
 def forward(self,e_ids,q_ids,disable_question=False):
  B=e_ids.shape[0]
  he=self.e(self.feat)[e_ids];hq=self.q(self.feat)[q_ids]
  A=F.softmax(self.a(he),-1);Bnode=F.softmax(self.b(he),-1);D=F.softmax(self.dir(he),-1)
  forward=Bnode[:,:,:,None]*A[:,:,None,:]
  back=A[:,:,:,None]*Bnode[:,:,None,:]
  G=torch.einsum('btij,btd->bdij',forward,D)+torch.einsum('btij,btd->bdij',back,D[:,:,self.op])
  s=F.softmax(self.src(hq),-1);t=F.softmax(self.tgt(hq),-1)
  cur=s[:,None,:];prev=torch.zeros(len(e_ids),device=e_ids.device)
  paths=[];z=[]
  for k in (1,2,3,4):
   cur=torch.einsum('bpi,bdij->bpdj',cur,G).reshape(len(e_ids),-1,8)
   p=torch.einsum('bpn,bn->bp',cur,t).clamp_min(1e-12)
   logp=F.log_softmax(p.log(),dim=-1);paths.append(logp)
   soft=F.softmax(p.log(),-1)
   entropy=-(soft*soft.clamp_min(1e-12).log()).sum(-1)/math.log(4**k)
   total=p.sum(-1);maximum=p.max(-1).values
   features=torch.stack([total.log1p(),maximum.log1p(),entropy,(total-prev).tanh(),maximum/(total+1e-7)],-1)
   h_input=torch.cat([torch.zeros_like(hq) if disable_question else hq,features],-1)
   z.append(self.stop(h_input).squeeze(-1));prev=total
  hz=torch.sigmoid(torch.stack(z[:3],dim=1)).clamp(1e-6,1-1e-6)
  logstop=torch.stack([hz[:,0],(1-hz[:,0])*hz[:,1],(1-hz[:,0])*(1-hz[:,1])*hz[:,2],(1-hz[:,0])*(1-hz[:,1])*(1-hz[:,2])],-1).clamp_min(1e-12).log()
  logits=torch.cat([paths[0]+logstop[:,0,None],paths[1]+logstop[:,1,None],paths[2]+logstop[:,2,None],paths[3]+logstop[:,3,None]],-1)
  entropy_reg=-(A*A.clamp_min(1e-8).log()).sum(-1).mean()-(Bnode*Bnode.clamp_min(1e-8).log()).sum(-1).mean()-(D*D.clamp_min(1e-8).log()).sum(-1).mean()
  balance=((A.mean((0,1))-1/8)**2).sum()+((Bnode.mean((0,1))-1/8)**2).sum()
  return logits,entropy_reg,balance

net=Joint();optim=torch.optim.AdamW(net.parameters(),lr=.0015,weight_decay=.0002);torch.set_num_threads(2)

def evaluate(data):
 net.eval();right=0;depths={k:[0,0] for k in (1,2,3,4)}
 with torch.no_grad():
  for chunk in torch.arange(len(data[2])).split(256):
   a,b,y,d=[r[chunk] for r in data];z,*_=net(a,b);pr=z.argmax(-1);c=(pr==y)
   right+=int(c.sum());
   for k in depths:
    mask=d==k;depths[k][0]+=int((c&mask).sum());depths[k][1]+=int(mask.sum())
 return right,{str(k):v for k,v in depths.items() if v[1]>0}

best=0;beststate=None;bestep=0;history=[]
for ep in range(1,121):
 net.train();p=torch.randperm(len(tr[2]));loss_sum=0
 for ii in p.split(256):
  a,b,y,d=[r[ii] for r in tr];z,entropy,balance=net(a,b)
  loss=F.nll_loss(z,y)+0.003*entropy+0.02*balance
  optim.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(net.parameters(),3);optim.step();loss_sum+=float(loss.detach())
 if ep==1 or ep%5==0:
  c,by=evaluate(v)
  history.append({'epoch':ep,'val_correct':c,'trainloss':round(loss_sum,3)})
  if c>best:best=c;bestep=ep;beststate={k:t.detach().clone() for k,t in net.state_dict().items()}
  if ep==1 or ep%10==0:print('EPOCH',ep,'val',c,'by_depth',by,'best',best,'sec',round(time.time()-st,1),flush=True)
  if best>=2990:break
net.load_state_dict(beststate)
idres=evaluate(te);ood=evaluate(t4)
print('FINAL unseen stories 1-3',idres,'unseen 4-hop',ood,'best epoch',bestep,'seconds',round(time.time()-st,1),flush=True)
name='joint_graph_halt_seed'+str(args.seed)
torch.save(net.state_dict(),O/(name+'.pt'))
(O/(name+'.json')).write_text(json.dumps({'seed':args.seed,'best_epoch':bestep,'validation_correct':best,'test_train_horizons':idres,'test_unseen_horizon_4':ood,'train_examples':len(train),'validation_examples':len(valid),'test_examples':len(test),'test_4_examples':len(test4),'history':history,'trainable_params':sum(p.numel() for p in net.parameters()),'pretrained_qwen_weights_updated':False,'description':'end-to-end training of continuous anonymous graph construction + shared STOP gate on final answers of length 1..3; all extra graph operations architectural; Qwen features fixed; OOD is length4 path generated from official held-out stories; all direction tokens are outputs; no intermediate labels','seconds':round(time.time()-st,2)},indent=2))
