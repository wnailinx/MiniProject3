"""Offline train/validation diagnostic labels, never imported by the agent."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parent
sys.path.insert(0,str(REPO/'MP3_SQL_student_package'))
from mp3_sql.common import load_bank,result_fingerprint,digest
from mp3_sql.score import load_cases
from mp3_sql.sqlite_tool import execute_sql

cli=argparse.ArgumentParser()
cli.add_argument('split',choices=('train','validation'))
args=cli.parse_args()
rows,manifest,folder=load_bank(args.split)
rows={r['id']:r for r in rows}
cases,case_root=load_cases(args.split,folder)
source=REPO/f'outputs/v13/features-{args.split}.jsonl'
tasks=[json.loads(s) for s in source.read_text(encoding='utf8').splitlines()]
seen=set()
for task in tasks:
    row=rows[task['id']]
    tables=[t['name'] for t in manifest['databases'][row['db_id']]['schema']]
    for candidate in task['candidates']:
        candidate['correct']=True
        for case in cases['tasks'][row['id']]:
            path=case_root/case['path']
            if case['database_sha256'] not in seen:
                assert digest(path)==case['database_sha256']
                seen.add(case['database_sha256'])
            result=execute_sql(path,candidate['sql'],tables,False)
            if not result['ok'] or result_fingerprint(result,row['ordered'])!=case['expected']:
                candidate['correct']=False
                break
output=ROOT/f'training/{args.split}-labelled.json'
output.write_text(json.dumps(tasks,ensure_ascii=False),encoding='utf8')
print(json.dumps({'split':args.split,'tasks':len(tasks),'candidates':sum(len(t['candidates']) for t in tasks),
                  'oracle':sum(any(c['correct'] for c in t['candidates']) for t in tasks),
                  'mixed':sum(len({c['correct'] for c in t['candidates']})==2 for t in tasks),
                  'source_sha256':digest(source),'output_sha256':digest(output)}),flush=True)
