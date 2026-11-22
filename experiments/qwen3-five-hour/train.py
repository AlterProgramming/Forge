#!/usr/bin/env python3
"""Continue a real Forge Qwen3-0.6B LoRA on GSM8K with a 5-hour compute budget.

The official GSM8K test split is sealed until one final comparison. Select
checkpoints only on disjoint training-split validation questions. Preserve
protocol competence by rehearsing the pre-existing Forge protocol.

Improvement claims require official heldout results, not decreasing train loss.
"""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import json
import random
import re
import shutil
import time
from pathlib import Path

ADAPTER = Path('models/qwen3-0.6b-communication-lora-v2')
EXPECTED_SHA = '4ea165d25f8e4e11f8a306ddee3d27558043051699e969cd13ad251db6e8143f'
A = ['maple','frost','ocean','cinder','violet','echo','pollen','tundra']
B = ['marble','orbit','fable','harbor','spruce','copper','velvet','comet']
SYS = 'Be accurate and concise. Follow the requested output format.'
PROTO_SYS = 'You exchange compact machine-readable messages. Reply with the exact required value and nothing else.'
END = re.compile(r'####\s*(-?\d[\d,]*)\s*(?:\n|$)')


def solution_number(text: str):
    m = list(END.finditer(text))
    return int(m[-1].group(1).replace(',', '')) if m else None


def prompt(q: str):
    return ('Solve this grade-school mathematics problem. Explain briefly, then '
            'finish with a separate line in the exact format #### <integer>.\n\n'
            'Problem: ' + q.strip() + '\nSolution:')


def proto(role, i, j):
    if role == 'ENC_X': return f'ROLE: ENCODE X\nVALUE: X={i}\nSYMBOL:', A[i]
    if role == 'ENC_Y': return f'ROLE: ENCODE Y\nVALUE: Y={j}\nSYMBOL:', B[j]
    if role == 'SEND': return f'ROLE: SEND\nOBSERVATION: X={i}; Y={j}\nMESSAGE:', A[i] + '|' + B[j]
    if role == 'RECEIVE': return f'ROLE: RECEIVE\nMESSAGE: {A[i]}|{B[j]}\nOUTPUT X AND Y:', f'X={i},Y={j}'
    if role == 'CALC': return f'ROLE: CALCULATE\nVALUES: X={i},Y={j}\nOPERATION: SUM\nANSWER:', str(i+j)
    raise ValueError(role)


def fingerprint(s): return hashlib.sha256(s.encode()).hexdigest()


def self_test():
    assert solution_number('work\n#### 1,204') == 1204
    assert solution_number('work\n#### -72\n') == -72
    assert solution_number('The answer is 72') is None
    assert solution_number('#### 5\n#### 8') == 8
    assert proto('RECEIVE', 2, 6)[1] == 'X=2,Y=6'
    assert len({(i,j) for i in range(8) for j in range(8) if (i+j)%4 == 0}) == 16
    assert '#### <integer>' in prompt('test')
    print('PASS: numeric parser, protocol mapping, immutable heldout split definitions')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--self-test',action='store_true')
    ap.add_argument('--out',default='qwen_5h_output')
    ap.add_argument('--model',default='Qwen/Qwen3-0.6B')
    ap.add_argument('--train-seconds',type=int,default=17400,
                    help='Five-hour job allocation leaves final scoring and artifact-write margin')
    ap.add_argument('--eval-period',type=int,default=900)
    ap.add_argument('--max-seq',type=int,default=320)
    ap.add_argument('--lr',type=float,default=8e-5)
    ap.add_argument('--seed',type=int,default=20261122)
    ap.add_argument('--val-count',type=int,default=48)
    ap.add_argument('--test-count',type=int,default=64)
    ap.add_argument('--rehearsal-interval',type=int,default=5)
    args=ap.parse_args()
    if args.self_test:
        self_test();return

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    from datasets import load_dataset

    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    rng=random.Random(args.seed)
    start=time.monotonic()
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    adapter_hash=hashlib.sha256((ADAPTER/'adapter_model.safetensors').read_bytes()).hexdigest()
    assert adapter_hash == EXPECTED_SHA, f'Original adapter hash mismatch: {adapter_hash}'
    info=dict(config=vars(args),started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
              source_adapter_sha256=adapter_hash, status='initializing',
              training=[],checkpoints=[],note='Original official test sealed until final; validation from train only.')
    def persist():
        (out/'results.json').write_text(json.dumps(info,indent=2))
    persist()

    print('Loading official Qwen base and *existing learned* Forge LoRA',flush=True)
    tok=AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    base=AutoModelForCausalLM.from_pretrained(args.model,dtype=torch.float32,low_cpu_mem_usage=True)
    base.config.use_cache=False
    net=PeftModel.from_pretrained(base,str(ADAPTER),is_trainable=True)
    trainables=[p for p in net.parameters() if p.requires_grad]
    assert sum(p.numel() for p in trainables)==393216,'Wrong adapter selected'
    opt=torch.optim.AdamW(trainables,lr=args.lr,weight_decay=0.0005)
    info['trainable_parameters']=sum(p.numel() for p in trainables)

    print('Loading established GSM8K public train/test splits',flush=True)
    train_raw=load_dataset('openai/gsm8k','main',split='train')
    indices=list(range(len(train_raw)))
    random.Random(args.seed).shuffle(indices)
    # Selection must be made without observing the official test labels or scores.
    val_ids=indices[:args.val_count]
    train_ids=indices[args.val_count:]
    info['data']=dict(name='openai/gsm8k',config='main',train_count=len(train_ids),
       validation_indices=val_ids,test_indices=test_ids,
       train_question_hash=fingerprint('\n'.join(train_raw[i]['question'] for i in train_ids)),
       validation_question_hash=fingerprint('\n'.join(train_raw[i]['question'] for i in val_ids)))

    def prefix(question,system=SYS):
        return tok.apply_chat_template([{'role':'system','content':system},
                 {'role':'user','content':question}],tokenize=True,
                 add_generation_prompt=True,enable_thinking=False)

    def seq_input(question,answer,system=SYS):
        pre=prefix(question,system)
        target=tok.encode(answer.strip(),add_special_tokens=False)+[tok.eos_token_id]
        if len(pre)+len(target)>args.max_seq:
            return None
        ids=pre+target
        labels=[-100]*len(pre)+target
        return torch.tensor(ids,dtype=torch.long).unsqueeze(0),torch.tensor(labels,dtype=torch.long).unsqueeze(0),len(target)

    def encode_gsm(row):
        # use the actual published answer including intermediate reasoning
        return seq_input(prompt(row['question']),row['answer'])

    train_examples=[(i,encode_gsm(train_raw[i])) for i in train_ids]
    train_examples=[(i,x) for i,x in train_examples if x is not None]
    val_examples=[(i,encode_gsm(train_raw[i])) for i in val_ids]
    val_examples=[(i,x) for i,x in val_examples if x is not None]
    if len(train_examples)<100 or len(val_examples)<16:
        raise RuntimeError(f'Insufficient GSM8K examples under max_seq={args.max_seq}: '+
                           str((len(train_examples),len(val_examples))))
    info['data']['train_usable']=len(train_examples)
    info['data']['val_usable']=len(val_examples)
    info['data']['length_filter']='prompt+official answer <= max_seq; same fixed filtering for train/validation'
    print('DATA',len(train_examples),len(val_examples),'official test still sealed',flush=True)

    def nll(data):
        net.eval()
        total=0.0; tokens=0
        with torch.no_grad():
            for _,item in data:
                ids,labs,ntok=item
                loss=net(input_ids=ids,labels=labs,use_cache=False).loss
                total+=float(loss)*ntok
                tokens+=ntok
        return total/max(tokens,1)

    def generate(question,system=SYS,max_new_tokens=128):
        pre=prefix(question,system)
        x=torch.tensor(pre,dtype=torch.long).unsqueeze(0)
        net.eval()
        with torch.inference_mode():
            y=net.generate(x,attention_mask=torch.ones_like(x),
                       max_new_tokens=max_new_tokens,do_sample=False,
                       pad_token_id=tok.eos_token_id)
        return tok.decode(y[0,len(pre):],skip_special_tokens=True).strip()

    comm_pairs=[(i,j) for i in range(8) for j in range(8) if (i+j)%4==0]
    def protocol_eval(pairs,roles=('ENC_X','ENC_Y','SEND','RECEIVE')):
        correct=0;rows=[]
        for i,j in pairs:
            for role in roles:
                question,expected=proto(role,i,j)
                raw=generate(question,PROTO_SYS,24)
                got=raw.splitlines()[0].strip() if raw else ''
                passed=(got.replace(' ','')==expected.replace(' ',''))
                correct+=int(passed)
                rows.append(dict(role=role,i=i,j=j,expected=expected,raw=raw,correct=passed))
        return dict(correct=correct,total=len(rows),traces=rows)

    # Validation only for checkpoint choice. Never use the official test here.
    initial_val=nll(val_examples)
    print('ORIGINAL_VALIDATION_NLL',initial_val,flush=True)
    info['baseline_validation_nll']=initial_val
    info['status']='training'
    best_val=initial_val
    best_step=0
    # Always persist the original checkpoint if no continuation wins.
    shutil.copytree(ADAPTER,out/'selected_adapter',dirs_exist_ok=True)
    persist()
    step=0
    last_eval=time.monotonic()
    try:
        while time.monotonic()-start<args.train_seconds:
            step+=1
            if step%args.rehearsal_interval==0:
                pair=(rng.randrange(8),rng.randrange(8))
                role=rng.choice(('ENC_X','ENC_Y','SEND','RECEIVE','CALC'))
                inp,expected=proto(role,*pair)
                item=seq_input(inp,expected,PROTO_SYS)
            else:
                item=rng.choice(train_examples)[1]
            if item is None:continue
            ids,labs,_=item
            net.train()
            opt.zero_grad(set_to_none=True)
            loss=net(input_ids=ids,labels=labs,use_cache=False).loss
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss at step '+str(step))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainables,1.0)
            opt.step()
            if step==1 or step%25==0:
                row=dict(step=step,train_loss=float(loss.detach()),
                         elapsed_seconds=round(time.monotonic()-start,1))
                info['training'].append(row)
                print('TRAIN',json.dumps(row),flush=True)
            if time.monotonic()-last_eval>=args.eval_period:
                score=nll(val_examples)
                improved=score<best_val-0.002
                if improved:
                    best_val=score;best_step=step
                    net.save_pretrained(out/'selected_adapter')
                row=dict(step=step,validation_nll=score,selected=improved,
                         best_validation_nll=best_val,elapsed_seconds=round(time.monotonic()-start,1))
                info['checkpoints'].append(row)
                info['best_step']=best_step
                print('CHECKPOINT',json.dumps(row),flush=True)
                persist()
                last_eval=time.monotonic()
        # Ensure a final checkpoint is considered without test access.
        score=nll(val_examples)
        if score<best_val-0.002:
            best_val=score;best_step=step
            net.save_pretrained(out/'selected_adapter')
        info['checkpoints'].append(dict(step=step,validation_nll=score,
                  selected=best_step==step,elapsed_seconds=round(time.monotonic()-start,1)))
        info['train_steps']=step
        info['train_elapsed_seconds']=round(time.monotonic()-start,1)
        info['selected_validation_nll']=best_val
        info['best_step']=best_step
        persist()
    except Exception as e:
        info['status']='interrupted'
        info['train_error']=repr(e)
        info['train_steps']=step
        persist()
        raise

    # Discard the last-step weights; evaluate the best VALIDATION-selected adapter.
    del opt,trainables
    del net
    del base
    import gc;gc.collect()
    base=AutoModelForCausalLM.from_pretrained(args.model,dtype=torch.float32,low_cpu_mem_usage=True).eval()
    base.config.use_cache=False
    net=PeftModel.from_pretrained(base,str(out/'selected_adapter'),is_trainable=False).eval()
    final_validation=nll(val_examples)
    assert abs(final_validation-best_val)<0.01,(final_validation,best_val)

    # Official test opened once, after selecting and freezing a candidate.
    test_raw=load_dataset('openai/gsm8k','main',split='test')
    test_ids=list(range(len(test_raw)))
    random.Random(args.seed+1).shuffle(test_ids)
    test_ids=test_ids[:args.test_count]
    info['data']['test_indices']=test_ids
    info['data']['test_question_hash']=fingerprint('\n'.join(test_raw[i]['question'] for i in test_ids))
    test_data=[(i,encode_gsm(test_raw[i])) for i in test_ids]
    test_data=[(i,x) for i,x in test_data if x is not None]
    info['data']['test_nll_usable']=len(test_data)
    assert len(test_data)>=16
    def eval_test():
        value=nll(test_data)
        rows=[]
        for i in test_ids:
            row=test_raw[i]
            got=generate(prompt(row['question']),max_new_tokens=128)
            parsed=solution_number(got)
            correct=parsed==solution_number(row['answer'])
            rows.append(dict(index=i,correct=correct,parsed=parsed,
                expected=solution_number(row['answer']),generated=got))
        return dict(token_nll=value,numeric_exact=sum(int(x['correct']) for x in rows),
                    count=len(rows),traces=rows)
    selected=eval_test()
    selected_proto=protocol_eval(comm_pairs)
    # Read original real checkpoint for the exact same official examples.
    del net;del base;gc.collect()
    base=AutoModelForCausalLM.from_pretrained(args.model,dtype=torch.float32,low_cpu_mem_usage=True).eval()
    base.config.use_cache=False
    net=PeftModel.from_pretrained(base,str(ADAPTER),is_trainable=False).eval()
    original=eval_test()
    original_proto=protocol_eval(comm_pairs)
    # Base Qwen control (LoRA disabled), original model and new adapter share all prompts.
    with net.disable_adapter():
        base_score=eval_test()
    improved=(best_step>0 and selected['token_nll']<original['token_nll']
       and selected['numeric_exact']>=original['numeric_exact']
       and selected_proto['correct']>=original_proto['correct']-1)
    info['final']=dict(selected=selected,original=original,base=base_score,
           selected_protocol=selected_proto,original_protocol=original_proto,
           # This is a narrow claim on a fixed public math protocol, not general intelligence.
           improvement_gate=improved,
           selection_rule='Best development NLL only; public test evaluated exactly once after selection',
           caveat='Only GSM8K-style math and Forge protocol measured. No general Qwen improvement implied.')
    info['selected_adapter_sha256']=hashlib.sha256((out/'selected_adapter'/'adapter_model.safetensors').read_bytes()).hexdigest()
    info['status']='completed'
    info['total_elapsed_seconds']=round(time.monotonic()-start,1)
    persist()
    (out/'README.md').write_text('''# Qwen3-0.6B 5-hour continuation\n\nBase: `Qwen/Qwen3-0.6B`; source: Forge saved LoRA v2.\nSelected adapter in `selected_adapter/`, which is the original if no validation checkpoint improved.\n`results.json` includes disjoint splits, hashes, checkpoints, traces, costs, and a\nstrict improvement gate. Only a true gate warrants a narrow math-domain improvement claim.\nLoad with `PeftModel.from_pretrained(base, "selected_adapter")`.\n''')
    print('FINAL',json.dumps(dict(step=step,best_step=best_step,
          selected_test_nll=selected['token_nll'],original_test_nll=original['token_nll'],
          selected_exact=selected['numeric_exact'],original_exact=original['numeric_exact'],
          selected_protocol=selected_proto['correct'],original_protocol=original_proto['correct'],
          improvement_gate=improved,elapsed=info['total_elapsed_seconds'])),flush=True)

if __name__=='__main__':main()
