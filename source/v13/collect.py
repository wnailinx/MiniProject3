"""Collect task-independent ranking features from own frozen train/validation traces."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parent
sys.path.insert(0,str(ROOT/'submission'))
from repair.terminal_rank import context,likelihood,pool,semantic_features

cli=argparse.ArgumentParser()
cli.add_argument('split',choices=('train','validation'))
args=cli.parse_args()
source=REPO/('outputs/v8/train-stats/train.json' if args.split=='train' else 'outputs/v10/final/validation.json')
if not source.exists() and args.split=='validation':source=REPO/'outputs/v10/full-terminal/validation.json'
rows={r['id']:r for r in map(json.loads,(REPO/f'MP3_SQL_student_package/mp3_sql/data/{args.split}.jsonl').read_text(encoding='utf8').splitlines())}
output=REPO/f'outputs/v13/features-{args.split}.jsonl'
output.parent.mkdir(parents=True,exist_ok=True)
assert not output.exists()
count=0
with output.open('w',encoding='utf8') as stream:
    for p in json.loads(source.read_text(encoding='utf8'))['predictions']:
        row=rows[p['id']]
        obs={'task':{'id':row['id'],'question':row['question']},'issue_sql':row['issue_sql'],
             'example':row['example'],'ordered':row['ordered'],'history':p['trace'][:-1],'remaining':8-p['tool_calls']}
        candidates,data=pool(obs['history']),context(obs)
        if len(candidates)<2 or data is None:continue
        neural=likelihood(data,candidates,p['id'])
        record={'id':p['id'],'db_id':row['db_id'],'incumbent':p['sql'],
                'candidates':[{'sql':s,'features':semantic_features(s,p['sql'],obs)|n} for s,n in zip(candidates,neural)]}
        stream.write(json.dumps(record,ensure_ascii=False)+'\n');stream.flush()
        count+=1
        if count%20==0: print(json.dumps({'split':args.split,'tasks':count}),flush=True)
print(json.dumps({'split':args.split,'tasks':count}),flush=True)
