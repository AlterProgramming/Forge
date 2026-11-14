#!/usr/bin/env python3
"""Differentiable anonymous graph workspace learned from final QA labels alone.

Official bAbI Task 19 (2-edge path finding). Six UNNAMED, learned node addresses;
no provided room-to-slot mapping, parser labels, supporting-fact indices,
intermediate graph targets, or text readback. Frozen Qwen3 event/question features.
Only explicit mathematical prior: path answers compose two relation edges,
with inverse of north/south and east/west (predefined direction semantics).
Compare exact classical reference and unstructured 1/2/4-hop neural baselines.
"""
import math,time,random,json,argparse
from pathlib import Path
from collections import Counter
import torch
from torch import nn
import torch.nn.functional as F
P=Path('/mnt/data/qwen_next')
a=argparse.ArgumentParser();a.add_argument('--epochs',type=int,default=160);a.add_argument('--seed',type=int,default=11819);a.add_argument('--nodes',type=int,default=6);a.add_argument('--dropout',type=float,default=0.05);a.add_argument('--reverse',type=str,default='learned',choices=['learned','fixed','none']);a.add_argument('--withhold',type=int,default=12);a.add_argument('--feature',type=str,default='qwen',choices=['qwen','random','permuted','whiten','centered']);args=a.parse_args()
random.seed(args.seed);torch.manual_seed(args.seed);torch.set_num_threads(2);start=time.monotonic()
record=torch.load(P/'qa19_frozen_qwen_sentence_embeddings.pt',weights_only=True);feat=record['feats'];vocab=record['vocab'];idx={x:i for i,x in enumerate(vocab)}


def parse(path):
 r=[];hist=[]
 for line in path.read_text().splitlines():
  n,s=line.split(' ',1)
  if n=='1':hist=[]
  if '\t' in s:
   q,ans,*_=s.split('\t');r.append((tuple(hist),q,ans))
  else:hist.append(s.strip())
 return r
full=parse(P/'qa19_10k_train.txt');test=parse(P/'qa19_10k_test.txt')
ans=sorted(set(q[2] for q in full));enc={x:i for i,x in enumerate(ans)}
D=['n','s','e','w'];opp=torch.tensor([1,0,3,2]);pairs=[(D.index(v.split(',')[0]),D.index(v.split(',')[1])) for v in ans]
assert len(ans)==12 and len(full)==10000 and len(test)==1000
spl=list(range(10000));random.Random(90117).shuffle(spl)
valid_ids=set(spl[:1000]);tr=[full[i] for i in range(10000) if i not in valid_ids];va=[full[i] for i in range(10000) if i in valid_ids]
# Select withheld event sentence types BEFORE any model training, independent of labels.
event_types=sorted(set(e for row in full for e in row[0]))
withheld=set(random.Random(1922).sample(event_types,args.withhold))
tr=[row for row in tr if not any(e in withheld for e in row[0])]
va=[row for row in va if not any(e in withheld for e in row[0])]
test_ood=[row for row in test if any(e in withheld for e in row[0])]
test_id=[row for row in test if all(e not in withheld for e in row[0])]
assert len(test_id)+len(test_ood)==len(test)
print('WITHHELD',args.withhold,'of',len(event_types),'TRAIN',len(tr),'VALID',len(va),'TEST_OOD',len(test_ood),'TEST_ID',len(test_id),flush=True)
# Fit all unsupervised representation geometry exclusively on lexical items occurring
# in training stories/questions, never on withheld test sentence features.
train_vocab_indices=sorted(set(idx[e] for row in tr for e in row[0])|set(idx[row[1]] for row in tr))
if args.feature=='random':feat=torch.randn(feat.shape,generator=torch.Generator().manual_seed(args.seed+620))*3.2
elif args.feature=='permuted':feat=feat[torch.randperm(len(feat),generator=torch.Generator().manual_seed(args.seed+620))]
elif args.feature=='centered':feat=feat-feat[train_vocab_indices].mean(0,keepdim=True)
elif args.feature=='whiten':
 orig=feat
 center=orig[train_vocab_indices].mean(0,keepdim=True)
 u,s,v=torch.linalg.svd(orig[train_vocab_indices]-center,full_matrices=False)
 rank=int((s>s[0]*1e-5).sum())
 feat=((orig-center)@v[:rank].T / s[:rank])@v[:rank]*8.0
 print('TRAIN-ONLY WHITEN rank',rank,'train_vocab',len(train_vocab_indices),'maxnorm_ood',round(float(feat[[idx[e] for e in withheld]].norm(dim=1).max()),3),flush=True)


def batches(dataset,batch=256,shuffle=False):
 order=list(range(len(dataset)))
 if shuffle:random.shuffle(order)
 for i in range(0,len(order),batch):
  x=[dataset[j] for j in order[i:i+batch]]
  e=torch.tensor([[idx[t] for t in r[0]] for r in x]);q=torch.tensor([idx[r[1]] for r in x]);y=torch.tensor([enc[r[2]] for r in x]);yield e,q,y
class LatentGraph(nn.Module):
 def __init__(self,n=6):
  super().__init__();self.n=n;self.register_buffer('feat',feat)
  self.e=nn.Sequential(nn.LayerNorm(1024),nn.Linear(1024,128),nn.GELU(),nn.Linear(128,96),nn.GELU())
  self.q=nn.Sequential(nn.LayerNorm(1024),nn.Linear(1024,128),nn.GELU(),nn.Linear(128,96),nn.GELU())
  self.a=nn.Linear(96,n);self.b=nn.Linear(96,n);self.dir=nn.Linear(96,4)
  self.src=nn.Linear(96,n);self.tgt=nn.Linear(96,n)
  self.prior=nn.Parameter(torch.zeros(12))
  self.rev_logits=nn.Parameter(torch.randn(4,4)*.1)
  self.dropout=nn.Dropout(args.dropout)
  self.register_buffer('pair_idx',torch.tensor([i*4+j for i,j in pairs]))
 def forward(self,events,query):
  # Encode each DISTINCT sentence once per batch to reuse static frozen-Qwen features.
  h=self.dropout(self.e(self.feat))[events]
  B,T,_=h.shape;N=self.n
  pa=F.softmax(self.a(h),-1);pb=F.softmax(self.b(h),-1);direction=F.softmax(self.dir(h),-1)
  # Event "A east of B" creates directed edge B --east--> A and reverse.
  forward=pb[:,:,:,None]*pa[:,:,None,:]
  backward=pa[:,:,:,None]*pb[:,:,None,:]
  G=torch.einsum('btij,btd->bdij',forward,direction)
  if args.reverse=='learned':reverse=direction@F.softmax(self.rev_logits,dim=-1)
  elif args.reverse=='fixed':reverse=direction[:,:,opp]
  else:reverse=torch.zeros_like(direction)
  G+=torch.einsum('btij,btd->bdij',backward,reverse)
  # Two differentiable rewrites: from source node -> intermediate -> target.
  qh=self.dropout(self.q(self.feat))[query]
  source=F.softmax(self.src(qh),-1)
  target=F.softmax(self.tgt(qh),-1)
  h1=torch.einsum('bi,bdij->bdj',source,G)
  h2=torch.einsum('bdj,bcjk->bdck',h1,G)
  score=torch.einsum('bdck,bk->bdc',h2,target).flatten(1)[:,self.pair_idx]
  logits=torch.log(score+1.e-7)+self.prior[None,:]*0.03
  entropy=-(pa*pa.clamp_min(1e-8).log()).sum(-1).mean()-(pb*pb.clamp_min(1e-8).log()).sum(-1).mean()
  entropy+=-(direction*direction.clamp_min(1e-8).log()).sum(-1).mean()
  # discourages trivial all-rooms-in-same-node solution
  node_balance=((pa.mean((0,1))-1/N)**2).sum()+((pb.mean((0,1))-1/N)**2).sum()
  return logits,entropy,node_balance
net=LatentGraph(args.nodes);opt=torch.optim.AdamW(net.parameters(),lr=.002,weight_decay=.0002)

def accuracy(rows,keep_vec=False):
 net.eval();c=0;N=0;vec=[];loss=0
 with torch.no_grad():
  for x,q,y in batches(rows):
   z,_,_=net(x,q);pred=z.argmax(-1);c+=int((pred==y).sum());N+=len(y)
   loss+=F.cross_entropy(z,y,reduction='sum').item()
   if keep_vec:vec+= (pred==y).int().tolist()
 return c,N,loss/N,vec
best=-1;state=None;epoch=None;history=[]
for ep in range(1,args.epochs+1):
 net.train();ll=[]
 for x,q,y in batches(tr,shuffle=True):
  logits,ent,balance=net(x,q)
  loss=F.cross_entropy(logits,y)+0.003*ent+0.02*balance
  opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(net.parameters(),3);opt.step();ll.append(float(loss.detach()))
 if ep==1 or ep%5==0:
  v=accuracy(va)
  if v[0]>best:best=v[0];epoch=ep;state={k:t.detach().clone() for k,t in net.state_dict().items()}
  history.append({'epoch':ep,'valid':v[0],'trainloss':round(sum(ll)/len(ll),4)})
  print('EPOCH',ep,'train',round(sum(ll)/len(ll),3),'valid',v[0],'best',best,'sec',round(time.monotonic()-start,1),flush=True)
  if best>=990:break
net.load_state_dict(state)
te=accuracy(test,True);oo=accuracy(test_ood,True);iid=accuracy(test_id,True)
print('OFFICIAL TEST',te[0],'/',te[1],'OOD',oo[0],'/',oo[1],'ID',iid[0],'/',iid[1],'best validation',best,'at',epoch,'seconds',round(time.monotonic()-start,1),flush=True)
name=f'qa19_anonymous_graph_ood{args.withhold}_{args.feature}_n{args.nodes}_reverse{args.reverse}_seed{args.seed}'
(P/(name+'.json')).write_text(json.dumps({'test':te[0],'n':te[1],'test_vec':te[3],'ood_correct':oo[0],'ood_n':oo[1],'ood_vec':oo[3],'id_correct':iid[0],'id_n':iid[1],'withheld_event_types':sorted(withheld),'train_count':len(tr),'val_count':len(va),'val':best,'best_epoch':epoch,'history':history,'seconds':round(time.monotonic()-start,1),'feature':args.feature,'reverse_mode':args.reverse,'node_count':args.nodes,'params':sum(x.numel() for x in net.parameters()),'method':'anonymous six-node continuous learned graph, latent edge binding and two-step tensor composition, pretrained frozen Qwen3 sentence embeddings, answer labels ONLY; structural prior: two graph edges + known reverse compass directions; no lexical parser or support supervision'},indent=2))
torch.save(net.state_dict(),P/(name+'.pt'))