from pathlib import Path
import torch,json
P=Path('/mnt/data/qwen_next')
s=(P/'qwen_last_layer_lora_halt.py').read_text().split('start_time=time.time();')[0]
exec(compile(s,str(P/'qwen_last_layer_lora_halt.py'),'exec'))

def complete(net,q,f,paths,y,d):
 net.eval();out=[];select=[];depth=[]
 with torch.no_grad():
  for ids in torch.arange(len(y)).split(96):
   p=net(q[ids],f[ids]).clamp_min(1e-12).log()
   s=paths[ids]
   logits=torch.cat([s[:,:4]+p[:,0,None],s[:,4:20]+p[:,1,None],s[:,20:84]+p[:,2,None],s[:,84:]+p[:,3,None]],1)
   pred=logits.argmax(-1)
   out.extend((pred==y[ids]).int().tolist())
   select.extend(pred.tolist())
   depth.extend((p.argmax(-1)+1).tolist())
 return {'correct':sum(out),'n':len(out),'vector':out,'selected_halt_depth':depth}

result={}
for v in [True,False]:
 net=QwenNativeStop(v);weights=torch.load(O/('lora_qwen_stop.pt' if v else 'frozen_qwen_stop.pt'),weights_only=True)
 net.load_state_dict(weights,strict=False)
 d={'id_full':complete(net,teQ,teF,teP,teY,teDepth),'ood4_full':complete(net,oQ,t4F,t4P,t4Y,t4D)}
 print(v,d['id_full']['correct'],d['ood4_full']['correct'],flush=True)
 result['qwen_last_layer_lora' if v else 'qwen_frozen_readout']=d
(O/'full_path_results.json').write_text(json.dumps(result))
