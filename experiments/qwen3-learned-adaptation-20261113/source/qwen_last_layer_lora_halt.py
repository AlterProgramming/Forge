#!/usr/bin/env python3
"""Experiment: actual low-rank weight update inside pretrained Qwen3 decoder last block.
Frozen Qwen3 base. Learn LoRA modifications of o_proj/down_proj and a STOP head
from final path answer probabilities. Continuous graph state; no text memory.
This is not full-model training or operation invention.
"""
import os,sys,time,copy,json,math,random
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from safetensors import safe_open
sys.path.insert(0,'/mnt/data/qwen3_env')
from transformers import AutoConfig
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer,Qwen3RotaryEmbedding
P=Path('/mnt/data/qwen_next');O=P/'qwen_native_lora_halt';O.mkdir(exist_ok=True)
# Import graph-based task generation and tensor stats from frozen control, without training it.
code=(P/'qwen_adaptive_halt_ood.py').read_text().split('class SharedStop(nn.Module):')[0]
exec(compile(code,str(P/'qwen_adaptive_halt_ood.py'),'exec'))
O=P/'qwen_native_lora_halt';O.mkdir(exist_ok=True)
# prepare features without graph network Q dimensions: replace 96-D task head with
# 1024-D frozen pretrained Qwen3 contextual embedding from question itself.
raw=torch.load(P/'qa19_frozen_qwen_sentence_embeddings.pt',weights_only=True)['feats']
qs=lambda rows:torch.tensor([idx[r[1]] for r in rows])
# Last dimension of previously computed trF is [96 task head, 5 graph-state features].
# Prior Qwen representations remain full float32 for use as fixed per-step input token.
trQ=raw[qs(train)];vaQ=raw[qs(valid)];teQ=raw[qs(test)];oQ=raw[qs(test4)]
mu=trF[:,:,-5:].reshape(-1,5).mean(0);std=trF[:,:,-5:].reshape(-1,5).std(0).clamp_min(.01)

class LoRA(nn.Module):
 def __init__(self,base,r=4,scale=4):
  super().__init__();self.base=base;self.r=r;self.scale=scale/r
  self.A=nn.Parameter(torch.randn(r,base.in_features)*0.003)
  self.B=nn.Parameter(torch.zeros(base.out_features,r))
  for p in self.base.parameters():p.requires_grad=False
 def forward(self,x):
  return self.base(x) + (F.linear(F.linear(x.float(),self.A),self.B)*self.scale).to(x.dtype)

class QwenNativeStop(nn.Module):
 def __init__(self,use_lora=True):
  super().__init__();cfg=AutoConfig.from_pretrained('/mnt/data/qwen3-local',local_files_only=True)
  cfg._attn_implementation='sdpa'
  self.block=Qwen3DecoderLayer(cfg,layer_idx=27)
  with safe_open('/mnt/data/qwen3-local/model.safetensors',framework='pt') as f:
   src={k[len('model.layers.27.'):]:f.get_tensor(k) for k in f.keys() if k.startswith('model.layers.27.')}
  self.block.load_state_dict(src,strict=True)
  self.block=self.block.to(torch.bfloat16)
  for p in self.block.parameters():p.requires_grad=False
  self.use_lora=use_lora
  if use_lora:
   self.block.self_attn.o_proj=LoRA(self.block.self_attn.o_proj)
   self.block.mlp.down_proj=LoRA(self.block.mlp.down_proj)
  self.rope=Qwen3RotaryEmbedding(cfg)
  g=torch.Generator().manual_seed(4137);W=torch.randn(1024,5,generator=g);W=W/W.norm(dim=0,keepdim=True)*45.0
  self.register_buffer('stat_projection',W)
  self.norm=nn.LayerNorm(1024)
  self.readout=nn.Sequential(nn.Linear(1024,64),nn.GELU(),nn.Linear(64,1))
 def stop_logits(self,qu,st):
  # 1 latent token from Qwen question and 1 token containing a Qwen-aligned,
  # edited continuous representation of graph reachability.
  scaled=((st-mu)/std).clamp(-5,5)
  other=qu+F.linear(scaled,self.stat_projection)
  inputs=torch.stack([qu,other],dim=1).to(torch.bfloat16)
  positions=torch.arange(2)[None,:].expand(len(qu),-1)
  emb=self.rope(inputs,positions)
  final=self.block(inputs,position_ids=positions,position_embeddings=emb,use_cache=False)
  logits=self.readout(self.norm(final[:,-1].float()))
  return logits[:,0]
 def forward(self,q,feats):
  B=q.shape[0]
  qq=q[:,None,:].expand(B,3,-1).reshape(B*3,-1)
  stats=feats[:,:3,-5:].reshape(B*3,5)
  logits=self.stop_logits(qq,stats).view(B,3)
  hz=torch.sigmoid(logits).clamp(1e-6,1-1e-6)
  p=torch.stack([hz[:,0],(1-hz[:,0])*hz[:,1],(1-hz[:,0])*(1-hz[:,1])*hz[:,2],(1-hz[:,0])*(1-hz[:,1])*(1-hz[:,2])],-1)
  return p

def evaluate(net,qu,stats,y,d,batch=96):
 net.eval();N=0;correct=0;correct_depth=0
 with torch.no_grad():
  for p in torch.arange(len(y)).split(batch):
   z=net(qu[p],stats[p]); k=z.argmax(-1)+1;correct_depth+=int((k==d[p]).sum());correct+=int((k==d[p]).sum());N+=len(p)
 return {'correct_depth':correct,'n':N,'depth_accuracy':round(correct_depth/N,4)}

def train(use_lora):
 torch.manual_seed(871+int(use_lora));net=QwenNativeStop(use_lora)
 params=[p for p in net.parameters() if p.requires_grad];print('TRAIN',use_lora,'parameters',sum(p.numel() for p in params),flush=True)
 opt=torch.optim.AdamW(params,lr=.001,weight_decay=.001)
 train_id=torch.randperm(len(trY))[:3600];best=-1;beststate=None;bestepoch=0;hist=[]
 for epoch in range(1,19):
  net.train();p=train_id[torch.randperm(len(train_id))];ll=[]
  for batch in p.split(48):
   probs=net(trQ[batch],trF[batch]); loss=F.nll_loss(probs.clamp_min(1e-8).log(),trDepth[batch]-1)
   opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(params,2);opt.step();ll.append(float(loss.detach()))
  if epoch==1 or epoch%2==0:
   with torch.no_grad():v=evaluate(net,vaQ[:800],vaF[:800],vaY[:800],vaDepth[:800])
   hist.append({'epoch':epoch,'valid800':v['correct_depth'],'loss':round(sum(ll)/len(ll),4),'seconds':round(time.time()-st,1)})
   if v['correct_depth']>best:best=v['correct_depth'];beststate=copy.deepcopy({k:x.detach().clone() for k,x in net.state_dict().items() if 'block.' not in k or '.A' in k or '.B' in k});bestepoch=epoch
   print('EPOCH',use_lora,epoch,'valid800',v['correct_depth'],'best',best,'seconds',round(time.time()-st,1),flush=True)
 # Restoring a subset of state dict only; full pretrained block remains fixed.
 net.load_state_dict(beststate,strict=False)
 test=evaluate(net,teQ,teF,teY,teDepth);ood=evaluate(net,oQ,t4F,t4Y,t4D)
 print('FINAL',use_lora,test,ood,flush=True)
 state={k:v.detach().cpu() for k,v in net.state_dict().items() if '.A' in k or '.B' in k or k.startswith('readout.') or k.startswith('norm.')}
 torch.save(state,O/('lora_qwen_stop.pt' if use_lora else 'frozen_qwen_stop.pt'))
 return {'trainable_params':sum(p.numel() for p in params),'best_epoch':bestepoch,'validation_best':best,'test_1to3':test,'test_unseen_4':ood,'curve':hist}

start_time=time.time();print('INIT',trF.shape,len(test4),flush=True)
with_lora=train(True)
no_lora=train(False)
(O/'results.json').write_text(json.dumps({'with_lora':with_lora,'no_lora':no_lora,'description':'Qwen3 last pretrained transformer decoder layer with rank-4 LoRA on attention output and MLP output weights, trainable readout. Earlier 27 layers fixed and not run; input is pretrained Qwen sentence final hidden vectors and fixed-projected latent graph state. This trains genuine Qwen3 decoder weight deltas but is not end-to-end model fine-tuning. Supervision comes from final path labels, which encode path length. Loss trains STOP length choice, not specific path quality. Holdout depth4 was never supervised.','time_seconds':round(time.time()-st,1)},indent=2))
print('SAVED',flush=True)
