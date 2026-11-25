"""Fit pairwise ranking on TRAIN; validation only chooses a global switching margin."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'submission'))
from repair.structure import key
from repair.terminal_rank import score

tasks=json.loads((ROOT/'training/train-labelled.json').read_text(encoding='utf8'))
validation=json.loads((ROOT/'training/validation-labelled.json').read_text(encoding='utf8'))
names=sorted({k for t in tasks for c in t['candidates'] for k in c['features']})
X=np.asarray([[c['features'].get(k,0.) for k in names] for t in tasks for c in t['candidates']])
y=np.asarray([c['correct'] for t in tasks for c in t['candidates']])
groups=np.asarray([i for i,t in enumerate(tasks) for c in t['candidates']])
fold=np.asarray([int(hashlib.sha256(t['db_id'].encode()).hexdigest()[:8],16)%4 for t in tasks for c in t['candidates']])


def fit(mask,reg):
    mean,scale=X[mask].mean(0),X[mask].std(0)
    scale=np.where(scale<1e-6,1.,scale)
    Z=np.clip((X-mean)/scale,-4,4)
    pairs=[]
    for g in np.unique(groups[mask]):
        pos=np.flatnonzero(mask&(groups==g)&y)
        neg=np.flatnonzero(mask&(groups==g)&~y)
        pairs.extend(Z[p]-Z[n] for p in pos for n in neg)
    D=np.asarray(pairs)
    w=np.zeros(len(names));m=w.copy();v=w.copy()
    for step in range(1,801):
        p=1/(1+np.exp(np.clip(D@w,-35,35)))
        grad=-(D.T@p)/len(D)+reg*w
        m=.9*m+.1*grad;v=.999*v+.001*grad**2
        w-=.025*(m/(1-.9**step))/(np.sqrt(v/(1-.999**step))+1e-8)
    return {'weights':dict(zip(names,w.tolist())),'mean':dict(zip(names,mean.tolist())),
            'scale':dict(zip(names,scale.tolist())),'pairs':len(D),'regularization':reg}


results=[]
for reg in (.01,.1,1.):
    scores=np.zeros(len(y))
    for f in range(4):
        model=fit(fold!=f,reg)
        scores[fold==f]=np.clip((X[fold==f]-np.asarray(list(model['mean'].values())))/np.asarray(list(model['scale'].values())),-4,4)@np.asarray(list(model['weights'].values()))
    n=sum(y[(idx:=np.flatnonzero(groups==g))[np.argmax(scores[idx])]] for g in np.unique(groups))
    results.append({'regularization':reg,'oof_correct':int(n)})
best=max(results,key=lambda x:(x['oof_correct'],x['regularization']))
model=fit(np.ones(len(y),dtype=bool),best['regularization'])
comparisons=[]
for margin in (0.,.25,.5,1.,2.,4.,8.):
    fixed=lost=changed=0
    for t in validation:
        candidates=t['candidates']
        old=next(i for i,c in enumerate(candidates) if key(c['sql'])==key(t['incumbent']))
        values=[score(c['features'],model) for c in candidates]
        choice=max(range(len(values)),key=lambda i:(values[i],i==old))
        if values[choice]-values[old]<=margin:choice=old
        changed+=choice!=old
        fixed+=candidates[choice]['correct'] and not candidates[old]['correct']
        lost+=not candidates[choice]['correct'] and candidates[old]['correct']
    comparisons.append({'margin':margin,'fixes':int(fixed),'regressions':int(lost),'changed':int(changed)})
chosen=max(comparisons,key=lambda x:(x['fixes']-x['regressions'],-x['regressions'],-x['changed'],-x['margin']))
model['margin']=chosen['margin']
(ROOT/'submission/terminal_rank.json').write_text(json.dumps(model,indent=2),encoding='utf8')
config=json.loads((ROOT/'submission/config.json').read_text())
config['rerank_margin']=chosen['margin']
(ROOT/'checkpoints/selected.json').write_text(json.dumps(config,indent=2),encoding='utf8')
record={'train_tasks':len(tasks),'features':len(names),'training_pairs':model['pairs'],
        'oof':results,'validation_margin_selection':comparisons,'selected':chosen,
        'lora_has_seen_train_targets':True,'test_labels_used':False}
(ROOT/'records/ranking-fit.json').write_text(json.dumps(record,indent=2),encoding='utf8')
print(json.dumps(record),flush=True)
