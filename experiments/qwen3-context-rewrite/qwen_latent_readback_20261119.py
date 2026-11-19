#!/usr/bin/env python3
"""Frozen-Qwen latent REWRITE/readback feasibility test. Outcome-only supervision.
Not a claim of general representation learning. CPU diagnostic, fixed seeds.
"""
import argparse,json,time,random,torch
from transformers import AutoTokenizer,AutoModelForCausalLM

def main():
 p=argparse.ArgumentParser()
 p.add_argument("--steps",type=int,default=12)
 p.add_argument("--output",default="latent_readback_results.json")
 a=p.parse_args();torch.manual_seed(119);random.seed(119);torch.set_num_threads(2)
 start=time.monotonic()
 tok=AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
 q=AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B",torch_dtype=torch.float32,low_cpu_mem_usage=True).eval()
 for w in q.parameters():w.requires_grad_(False)
 dim=q.config.hidden_size
 # The same trainable parameters for both arms; blind arm replaces previous state with zero.
 class Controller(torch.nn.Module):
  def __init__(self):
   super().__init__();self.update=torch.nn.Sequential(torch.nn.Linear(dim*2,64),torch.nn.Tanh(),torch.nn.Linear(64,dim))
   self.read=torch.nn.Linear(dim,2)
  def forward(self,events,blind=False,ablate=False):
   state=torch.zeros(1,dim)
   for e in events:
    event=q.get_input_embeddings()(torch.tensor([[e]])).detach().reshape(1,dim)
    prev=state.detach()*0 if blind else state
    state=state+self.update(torch.cat([prev,event],-1))
    # Qwen must consume updated state; the next step uses Qwen's readback as input.
    inp=state.unsqueeze(1)
    out=q(inputs_embeds=inp,output_hidden_states=True,use_cache=False,return_dict=True)
    state=out.hidden_states[-1][:,-1,:]
    if ablate:state=state.detach()*0
   return self.read(state)
 def dataset(n,seed,task,length):
  rng=random.Random(seed);rows=[]
  for _ in range(n):
   bits=[rng.randrange(2) for _ in range(length)]
   # Independent tasks: parity and final-write, no intermediate labels
   target=(sum(bits)%2) if task=="parity" else bits[-1]
   events=[tok.encode("A" if b else "B",add_special_tokens=False)[0] for b in bits]
   rows.append((events,target))
  return rows
 results={}
 for task in ("parity","last_write"):
  for blind in (False,True):
   model=Controller();optim=torch.optim.AdamW(model.parameters(),lr=2e-4)
   train=dataset(48,100,task,3)
   losses=[]
   for i in range(a.steps):
    ev,y=train[i%len(train)];optim.zero_grad()
    logits=model(ev,blind=blind)
    loss=torch.nn.functional.cross_entropy(logits,torch.tensor([y]))
    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1)
    optim.step();losses.append(float(loss.detach()))
   def evaluate(length,ablate=False):
    test=dataset(24,10000+length,task,length);right=0
    with torch.no_grad():
     for ev,y in test:right+=int(model(ev,blind=blind,ablate=ablate).argmax(-1).item()==y)
    return right
   results[f"{task}/blind={blind}"]={"train_loss_first":losses[0],"train_loss_last":losses[-1],
      "test_len3_correct_24":evaluate(3),"test_len6_correct_24":evaluate(6),
      "ablate_len6_correct_24":evaluate(6,True),
      "trainable_parameters":sum(p.numel() for p in model.parameters())}
   print(task,blind,results[f"{task}/blind={blind}"],flush=True)
 payload={"config":vars(a),"seed":119,"train_seed":100,"test_seed_rule":"10000+length",
 "tasks":["parity","last_write"],"train_length":3,"train_examples":48,"test_examples":24,
 "qwen":"Qwen/Qwen3-0.6B","frozen":True,"results":results,
 "wall_seconds":time.monotonic()-start,
 "limitations":"Tiny outcome-only CPU feasibility probe; Qwen is frozen; update is a trainable MLP; no proof of generality. Training and inference use full Qwen forward at each step; ablation zeroes readback, not a matched-FLOP intervention."}
 open(a.output,"w").write(json.dumps(payload,indent=2))
if __name__=="__main__":main()
