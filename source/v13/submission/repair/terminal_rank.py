"""Rank actual public-matching checks, using semantic and conditional LM evidence."""
from difflib import SequenceMatcher
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import time

from sqlglot import exp
from .obligations import Request, signals
from .rewriter import BASE_REVISION, load_model, messages, payload
from .structure import Schema, header_agreement, invalid_bindings, key, parse, parts


def pool(history):
    result = {}
    for event in history:
        if event['tool'] == 'check' and event['observation'].get('matches_example'):
            result.setdefault(key(event['sql']), event['sql'])
    return list(result.values())


def context(observation):
    history = observation['history']
    schema = next((e['observation'] for e in history if e['tool'] == 'schema'), None)
    if not schema:
        return None
    checks = [e for e in history if e['tool'] == 'check']
    return payload(observation['task']['question'], observation['issue_sql'], observation['example'],
                   observation['ordered'], schema, checks)


def semantic_features(sql, incumbent, observation):
    history = observation['history']
    records = next(e['observation'] for e in history if e['tool'] == 'schema')
    schema, tree = Schema(records), parse(sql)
    top = tree if isinstance(tree, exp.Select) else tree.find(exp.Select)
    question, example = observation['task']['question'], observation['example']
    identity, original = key(sql), key(observation['issue_sql'])
    hints = [s for e in history if e['tool'] == 'repair_hint' for s in e['observation']]
    words = parts(question)
    columns = list(tree.find_all(exp.Column))
    tables = list(tree.find_all(exp.Table))
    f = {'incumbent': int(identity == key(incumbent)), 'original': int(identity == original),
         'empty': int(not example['row_count']), 'width': header_agreement(top, example['columns']),
         'bad_bindings': min(5, invalid_bindings(tree, schema)),
         'hint_support': sum(key(h) == identity for h in hints)/4,
         'edit_distance': 1-SequenceMatcher(None,original,identity).ratio(),
         'size': math.log1p(len(identity)), 'tables': len(tables)/5,
         'question_columns': sum(bool(parts(c.name) & words) for c in columns)/max(1,len(columns)),
         'question_tables': sum(bool(parts(t.name) & words) for t in tables)/max(1,len(tables))}
    for name in ('distinct','group','having','order','limit','where'):
        f['has:'+name] = int(bool(top.args.get(name)))
    for cls in (exp.Select,exp.Join,exp.Count,exp.Min,exp.Max,exp.Avg,exp.Sum,exp.Not,
                exp.In,exp.Exists,exp.Like,exp.Union,exp.Intersect,exp.Except,exp.Literal):
        f['nodes:'+cls.__name__] = min(8,len(list(tree.find_all(cls))))/4
    f.update({'signal:'+k:float(v) for k,v in signals(sql,Request.read(question),example,schema).items()})
    return f


def likelihood(data, candidates, task_id):
    """Teacher-forced forward passes; these do not generate new SQL/completions."""
    adapter, device = os.environ['MP3_ADAPTER_DIR'], os.environ.get('MP3_MODEL_DEVICE','cpu')
    torch, tokenizer, model = load_model(adapter,device)
    prompt = tokenizer.apply_chat_template(messages(data),tokenize=False,add_generation_prompt=True,
                                            enable_thinking=False)
    prefix = tokenizer.encode(prompt,add_special_tokens=False)
    result = []
    started = time.perf_counter()
    for sql in candidates:
        target = tokenizer.encode(sql.strip()+tokenizer.eos_token,add_special_tokens=False)
        ids = torch.tensor([prefix+target],device=device)
        with torch.inference_mode():
            logits = model(input_ids=ids,logits_to_keep=len(target)+1,use_cache=False).logits[0,:-1].float()
            values = logits.log_softmax(-1).gather(1,torch.tensor(target,device=device)[:,None]).squeeze(1)
            values = values.cpu().tolist()
        ordered = sorted(values)
        f = {'lm_mean':sum(values)/len(values), 'lm_sum':sum(values)/100,
             'lm_worst_quarter':sum(ordered[:max(1,len(values)//4)])/max(1,len(values)//4),
             'lm_length':math.log1p(len(values))}
        result.append(f)
        del logits,ids
    record = {'id':task_id,'kind':'conditional_scoring','prompt':prompt,'candidates':candidates,
              'features':result,'seconds':time.perf_counter()-started,'model':'Qwen/Qwen3-0.6B',
              'base_revision':BASE_REVISION,'adapter':adapter,'device':device,'new_completions':0}
    if os.environ.get('MP3_RANK_LOG'):
        with Path(os.environ['MP3_RANK_LOG']).open('a',encoding='utf8') as stream:
            stream.write(json.dumps(record,ensure_ascii=False)+'\n')
    return result


@lru_cache(maxsize=1)
def weights():
    return json.loads((Path(__file__).resolve().parent.parent/'terminal_rank.json').read_text(encoding='utf8'))


def score(features, model):
    return sum(w*max(-4.,min(4.,(features.get(k,0.)-model['mean'][k])/model['scale'][k]))
               for k,w in model['weights'].items())


def select(observation, incumbent, config):
    candidates, data = pool(observation['history']), context(observation)
    if len(candidates)<2 or data is None:
        return incumbent
    neural = likelihood(data,candidates,observation['task']['id'])
    features = [semantic_features(sql,incumbent,observation)|n for sql,n in zip(candidates,neural)]
    model = weights()
    values = [score(f,model) for f in features]
    old = next(i for i,s in enumerate(candidates) if key(s)==key(incumbent))
    best = max(range(len(values)),key=lambda i:(values[i],i==old))
    margin = config.get('rerank_margin',model.get('margin',0.))
    return candidates[best] if values[best]-values[old] > margin else incumbent
