#!/usr/bin/env python3
"""Outcome-trained Qwen-conditioned selector of 1/2/3 relational computations.

Uses a frozen Qwen3-0.6B-derived graph checkpoint. The selector receives Qwen
question features and latent graph reachability statistics, then learns which
of three *supplied* traversal operators to invoke. It is trained on final answer
likelihood, not instruction/operator labels or intermediate reasoning traces.

Task family: held-out official bAbI-19 stories, with additional shortest-path
queries of length 1 and 3 constructed from their five visible graph edges.
This is an exploratory mixed-horizon extension, not an independent benchmark.
"""
import time, json, math, random, re, copy, argparse, collections
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

P=Path('/mnt/data/qwen_next'); O=P/'qwen_operator_autoselect'; O.mkdir(exist_ok=True)
parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,default=4471);parser.add_argument('--epochs',type=int,default=120);parser.add_argument('--checkpoint',type=str,default='qa19_minimal_adapter_compose_anytime_whiten_n8_seed11820.pt');args=parser.parse_args()
torch.set_num_threads(2);torch.manual_seed(args.seed);random.seed(args.seed);st=time.time()
ck=torch.load(P/args.checkpoint,weights_only=True)
feat=ck['feat'];index=torch.load(P/'qa19_frozen_qwen_sentence_embeddings.pt',weights_only=True)['vocab'];idx={x:i for i,x in enumerate(index)}
D=['n','s','e','w']; opp={'n':'s','s':'n','e':'w','w':'e'};words={'north':'n','south':'s','east':'e','west':'w'}
pat=re.compile(r'The (\w+) is (north|south|east|west) of the (\w+)\.')
ques=re.compile(r'How do you go from the (\w+) to the (\w+)\?')

# All tasks are drawn from the original complete graph. The exact path-finding
# routine is *only* used to label synthetic training/test questions; it never
# supplies the neural graph, router, or intermediate training supervision.
def parse(path):
 rows=[];hist=[];sid=-1
 for line in path.read_text().splitlines():
  n,s=line.split(' ',1)
  if n=='1':hist=[];sid+=1
  if '\t' in s:
   q,a,*_=s.split('\t');rows.append({'sid':sid,'hist':tuple(hist),'q':q.strip(),'a':a.strip()})
  else:hist.append(s.strip())
 return rows

def exact_graph(hist):
 edges=collections.defaultdict(list)
 for s in hist:
  m=pat.fullmatch(s)
  assert m,s
  u,di,v=m.groups();d=words[di]
  edges[v].append((u,d));edges[u].append((v,opp[d]))
 assert len(edges)==6
 return edges

def path(g,src,dst):
 q=collections.deque([(src,())]);seen=set()
 while q:
  u,r=q.popleft()
  if u==dst:return r
  if u in seen:continue
  seen.add(u)
  q.extend((v,r+(d,)) for v,d in g[u] if v not in seen)
 raise ValueError('No path')

def enrich(row,which):
 g=exact_graph(row['hist']); pair=ques.fullmatch(row['q']); assert pair
 original=path(g,*pair.groups());assert tuple(row['a'].split(','))==original and len(original)==2
 all_by_length={k:[] for k in (1,2,3)}
 for a in sorted(g):
  for b in sorted(g):
   if a==b:continue
   steps=path(g,a,b)
   if len(steps) in all_by_length:all_by_length[len(steps)].append((a,b,steps))
 rng=random.Random(args.seed*1000003+row['sid']+which*131071)
 result=[]
 for k in (1,2,3):
  a,b,steps=rng.choice(all_by_length[k])
  q=f'How do you go from the {a} to the {b}?'
  assert q in idx
  result.append((row['hist'],q,steps,k,row['sid']))
 return result

def fixed_projection(feat,prefix):
 h=F.layer_norm(feat,(1024,),weight=ck[prefix+'.0.weight'],bias=ck[prefix+'.0.bias'])
 h=F.gelu(F.linear(h,ck[prefix+'.1.weight'],ck[prefix+'.1.bias']))
 h=F.gelu(F.linear(h,ck[prefix+'.3.weight'],ck[prefix+'.3.bias']))
 return h

def lin(h,prefix):return F.linear(h,ck[prefix+'.weight'],ck[prefix+'.bias'])

E=fixed_projection(feat,'e');Q=fixed_projection(feat,'q')
pa=F.softmax(lin(E,'a'),-1);pb=F.softmax(lin(E,'b'),-1);dr=F.softmax(lin(E,'dir'),-1)
src=F.softmax(lin(Q,'src'),-1);tgt=F.softmax(lin(Q,'tgt'),-1)
N=pa.shape[-1];opp_t=torch.tensor([1,0,3,2]);
all_event=pb[:,:,None]*pa[:,None,:];reverse=pa[:,:,None]*pb[:,None,:]
graph_event=dr[:,:,None,None]*all_event[:,None,:,:]+dr[:,opp_t,None,None]*reverse[:,None,:,:]
train_rows=parse(P/'qa19_10k_train.txt');test_rows=parse(P/'qa19_10k_test.txt')
assert len(train_rows)==10000 and len(test_rows)==1000
ids=list(range(10000));random.Random(90117).shuffle(ids);va=set(ids[:1000]);
train=[it for r in train_rows if r['sid'] not in va for it in enrich(r,0)]
valid=[it for r in train_rows if r['sid'] in va for it in enrich(r,0)]
test=[it for r in test_rows for it in enrich(r,1)]
print('DATA',len(train),len(valid),len(test),'graph nodes',N,'vocab',len(index),flush=True)

# Features for route selection are computed with the frozen upstream graph.
# No trainable part can see the ground truth label except via final CE.
@torch.no_grad()
def precompute(rows,batch=512):
 inputs=[]; labels=[]; details=[]; depth=[]; upper=[]
 for j in range(0,len(rows),batch):
  chunk=rows[j:j+batch];qq=torch.tensor([idx[r[1]] for r in chunk]);ev=torch.tensor([[idx[s] for s in r[0]] for r in chunk])
  G=graph_event[ev].sum(1);ss=src[qq];tt=tgt[qq]
  states=ss[:,None,:]; features=[];raws=[]
  for k in (1,2,3):
   states=torch.einsum('bpi,bdij->bpdj',states,G).reshape(len(chunk),-1,N)
   path_weights=torch.einsum('bpn,bn->bp',states,tt).clamp_min(1e-12)
   lp=path_weights.log(); cond=F.log_softmax(lp,dim=-1)
   raws.append(cond)
   ps=F.softmax(lp,-1)
   entropy=-(ps*ps.clamp_min(1e-12).log()).sum(-1)/math.log(4**k)
   # The reachability of target at each horizon is useful before selecting length.
   features.extend([path_weights.sum(-1).log1p(),path_weights.max(-1).values.log1p(),entropy])
  stats=torch.stack(features,dim=1)
  # Router sees pre-trained Qwen query features + summaries of three latent computations.
  x=torch.cat([Q[qq],stats],dim=-1)
  pred=torch.cat(raws,dim=-1)
  y=[]
  for r in chunk:
   route=r[2];k=len(route);code=0
   for d in route:code=code*4+D.index(d)
   offset={1:0,2:4,3:20}[k]
   y.append(offset+code)
  y=torch.tensor(y)
  details.append(pred);inputs.append(x);labels.append(y);depth.extend(r[3] for r in chunk)
 return torch.cat(inputs),torch.cat(details),torch.cat(labels),torch.tensor(depth)
trX,trScore,trY,trDepth=precompute(train);vaX,vaScore,vaY,vaDepth=precompute(valid);teX,teScore,teY,teDepth=precompute(test)
print('CACHED latent graph computations',time.time()-st,flush=True)
# Determine if supplied frozen graph already solves the per-horizon subproblems.
def hop_summary(scores,y,depth):
 t={}
 for k,(start,end) in {1:(0,4),2:(4,20),3:(20,84)}.items():
  mask=depth==k
  pp=scores[mask,start:end].argmax(-1)+start
  t[str(k)]={'n':int(mask.sum()),'oracle_depth_correct':int((pp==y[mask]).sum())}
 return t
baseline=hop_summary(teScore,teY,teDepth);print('HOP-OPERATOR capacity',baseline,flush=True)

class Router(nn.Module):
 def __init__(self,feature_dim,graph_stats=True):
  super().__init__();self.graph_stats=graph_stats
  self.l=nn.Sequential(nn.LayerNorm(feature_dim),nn.Linear(feature_dim,64),nn.GELU(),nn.Linear(64,3))
 def forward(self,x):
  if not self.graph_stats:
   x=torch.cat((x[:,:-9],torch.zeros_like(x[:,-9:])),dim=-1)
  return self.l(x)

def combine(router,x,paths):
 depth=F.log_softmax(router(x),-1)
 return torch.cat([paths[:,:4]+depth[:,0,None],paths[:,4:20]+depth[:,1,None],paths[:,20:]+depth[:,2,None]],dim=-1)

def evaluate(router,x,paths,y,depth):
 router.eval()
 with torch.no_grad():
  z=combine(router,x,paths);p=z.argmax(-1);acc=(p==y).float().mean().item();d=router(x).argmax(-1)+1
  return {'accuracy':round(acc,6),'correct':int((p==y).sum()),'n':len(y),'depth_accuracy':round((d==depth).float().mean().item(),6),'correct_by_depth':{str(k):int(((p==y)&(depth==k)).sum()) for k in (1,2,3)},'selected_depth_distribution':{str(k):int((d==k).sum()) for k in (1,2,3)}},p,d

def train_router(graph_stats=True):
 torch.manual_seed(args.seed+int(graph_stats));net=Router(trX.shape[-1],graph_stats=graph_stats);opt=torch.optim.AdamW(net.parameters(),lr=.002,weight_decay=.001)
 best=-1;best_state=None;best_ep=None;hist=[]
 for ep in range(1,args.epochs+1):
  net.train();order=torch.randperm(len(trY));ll=0
  for ii in order.split(256):
   logits=combine(net,trX[ii],trScore[ii]);loss=F.nll_loss(logits,trY[ii]);opt.zero_grad(set_to_none=True);loss.backward();opt.step();ll+=loss.item()
  if ep==1 or ep%5==0:
   score,_p,_d=evaluate(net,vaX,vaScore,vaY,vaDepth)
   hist.append({'epoch':ep,'val':score['correct'],'depth':score['depth_accuracy']})
   if score['correct']>best:best=score['correct'];best_ep=ep;best_state=copy.deepcopy(net.state_dict())
   if ep%10==0 or ep==1:print('ROUTER','graph' if graph_stats else 'no_graph',ep,'valid',score,'sec',round(time.time()-st,1),flush=True)
 net.load_state_dict(best_state)
 score,pp,dd=evaluate(net,teX,teScore,teY,teDepth)
 print('OFFICIAL held-out stories mixed-horizon',graph_stats,score,'best_epoch',best_ep,flush=True)
 return net,score,pp,dd,best_ep,hist

learned,yes,pred,dec,ep,hist=train_router(True)
nograph,no,pn,dn,en,hn=train_router(False)
# The frozen no-choice comparison always applies one fixed operator (best possible on validation).
fixed={}
for k,(a,b) in {1:(0,4),2:(4,20),3:(20,84)}.items():
 z=teScore[:,a:b].argmax(-1)+a
 fixed[str(k)]={'correct':int((z==teY).sum()),'n':len(teY)}
print('FIXED',fixed,'total sec',time.time()-st,flush=True)
result={'description':'Qwen-frozen graph, task-conditioned learned selection over 1, 2, or 3 supplied relational traversal operators using final answer loss. Mixed-horizon shortest paths generated deterministically from official bAbI Task 19 tree stories. No intermediate hop labels, but final answer string length reveals hop count. Graph pretrained using 2-hop official Task 19 answer labels. All Qwen weights frozen. This is adaptive selection, NOT invention of new operators nor end-to-end Qwen fine-tuning.',
 'seed':args.seed,'feature_checkpoint':args.checkpoint,'train_examples':len(train),'validation_examples':len(valid),'test_examples':len(test),'official_train_stories':9000,'official_validation_stories':1000,'official_test_stories':1000,'hops_capacity':baseline,'graph_aware_router':yes,'graph_blind_router':no,'fixed_operators':fixed,'router_best_epoch':ep,'graph_blind_best_epoch':en,'training_curve':hist,'graph_blind_curve':hn,'seconds':round(time.time()-st,2),'test_correct_vector':(pred==teY).int().tolist(),'test_blind_correct_vector':(pn==teY).int().tolist(),'test_depth_truth':teDepth.tolist(),'test_depth_select':dec.tolist(),'test_depth_blind':dn.tolist(),'notes':['Architecture and graph operator menu provided; only operator selection trained','Vocab of sentence/query types shared train/test','Official bAbI Path Finding has only 2-hop answers, so extended 1,3 hop questions are derived using exact BFS; do not report as official bAbI scores','Train/test are disjoint stories but official test has been seen in earlier exploratory experiments','The output answer length discloses number of hops; routing objective thereby has implicit hop supervision even without separate labels']}
(O/'results.json').write_text(json.dumps(result,indent=2))
torch.save({'model':learned.state_dict(),'feature_checkpoint':args.checkpoint,'input_dim':trX.shape[-1]},O/'router.pt')
torch.save({'model':nograph.state_dict(),'feature_checkpoint':args.checkpoint,'input_dim':trX.shape[-1]},O/'blind_router.pt')
print('SAVED',O/'results.json',flush=True)
