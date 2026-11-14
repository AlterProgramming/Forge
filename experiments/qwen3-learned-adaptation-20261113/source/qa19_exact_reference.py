#!/usr/bin/env python3
"""Independent exact classical graph reference for official bAbI Task 19.
Explicit graph construction; not neural, not learned.
"""
from pathlib import Path
from collections import deque
import re,json,time
P=Path('/mnt/data/qwen_next')
pat=re.compile(r'^The (\w+) is (north|south|east|west) of the (\w+)\.$')
qpat=re.compile(r'^How do you go from the (\w+) to the (\w+)\?$')
vec={'north':(0,1),'south':(0,-1),'east':(1,0),'west':(-1,0)}
rev={'north':'south','south':'north','east':'west','west':'east'}
short={'north':'n','south':'s','east':'e','west':'w'}
records=[];hist=[]
for line in (P/'qa19_10k_test.txt').read_text().splitlines():
 number,s=line.split(' ',1)
 if number=='1':hist=[]
 if '\t' in s:
  q,a,*refs=s.split('\t');records.append((hist[:],q,a))
 else:hist.append(s.strip())
start=time.monotonic();n=0;wrong=[];unresolved=[];alldist=[]
for events,q,a in records:
 edges={}
 for ev in events:
  x,d,y=pat.fullmatch(ev).groups()
  edges.setdefault(y,[]).append((x,short[d]))
  edges.setdefault(x,[]).append((y,short[rev[d]]))
 source,target=qpat.fullmatch(q).groups()
 stack=deque([(source,'')]);seen={source};got=None
 while stack:
  u,path=stack.popleft()
  if u==target:got=','.join(path);break
  for v,d in edges.get(u,[]):
   if v not in seen:seen.add(v);stack.append((v,path+d))
 if got==a:n+=1
 else:wrong.append({'q':q,'correct':a,'derived':got,'facts':events})
print('CLASSICAL REFERENCE',n,'/',len(records),'bad',wrong[:2],'elapsed',round(time.monotonic()-start,3))
(P/'qa19_classical_reference_results.json').write_text(json.dumps({'correct':n,'total':len(records),'wrong':wrong[:10],'description':'Handwritten exact graph shortest-path reference; trained on no QA examples'},indent=2))
