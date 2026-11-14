#!/usr/bin/env python3
"""Inference-only loader for the exact trained Qwen3-connected, joint graph+stop adapter.

This is NOT a general language generation LoRA or a drop-in modification of
Qwen3-0.6B's transformer weights. It consumes integer IDs for the 150 trained
bAbI Path Finding event and question sentences. The checkpoint embeds its own
frozen Qwen feature matrix. The pretrained Qwen3 model is not loaded at runtime.
"""
import json,math,argparse
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
feat=torch.zeros(150,1024)  # replaced by 'feat' buffer in the checkpoint

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

DIRECTIONS=['north','south','east','west']

def load_adapter(checkpoint_path):
    model=Joint()
    state=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
    model.load_state_dict(state,strict=True)
    model.eval()
    return model

def decode_path(prediction):
    pred=int(prediction)
    if pred<4: k=1; code=pred
    elif pred<20: k=2; code=pred-4
    elif pred<84: k=3; code=pred-20
    else: k=4; code=pred-84
    out=[]
    for _ in range(k):
        out.append(DIRECTIONS[code%4]);code//=4
    return list(reversed(out))

def predict(model, vocab, events, question):
    idx={str(s):i for i,s in enumerate(vocab)}
    e_ids=torch.tensor([[idx[s] for s in events]],dtype=torch.long)
    q_ids=torch.tensor([idx[question]],dtype=torch.long)
    with torch.no_grad():
        logprobs,*_=model(e_ids,q_ids)
        top=int(logprobs[0].argmax().item())
    return {'answer':','.join(decode_path(top)),'output_index':top,'log_probability':float(logprobs[0,top].item())}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Run exact bAbI-path-trained Qwen3 adaptation checkpoint')
    parser.add_argument('--weights',default='checkpoints/joint_graph_halt_curriculum_seed61426.pt')
    parser.add_argument('--vocab',default='vocab_150.json')
    parser.add_argument('--input',help='JSON with events:[strings], question:string')
    args=parser.parse_args()
    model=load_adapter(args.weights)
    vocab=json.loads(Path(args.vocab).read_text())
    if args.input:
        inp=json.loads(Path(args.input).read_text())
        print(json.dumps(predict(model,vocab,inp['events'],inp['question']),indent=2))
    else:
        print(f'Loaded learned Qwen3-connected adapter: {sum(p.numel() for p in model.parameters()):,} trainable parameter tensors; {len(vocab)} sentence/question forms.')
