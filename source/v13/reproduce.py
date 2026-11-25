"""Reproduce an extracted source/adapter ZIP with an external official package."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path.insert(0,str(ROOT))
from evaluate import fingerprints

def read(path):
    return json.loads(path.read_text(encoding="utf-8"))
def lines(path):
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []

cli = argparse.ArgumentParser()
cli.add_argument("--official-package",type=Path,required=True)
cli.add_argument("--adapter",type=Path,required=True)
cli.add_argument("--reference",type=Path,default=REPO / "outputs")
cli.add_argument("--name",default="independent-reproduction")
args = cli.parse_args()
freeze = read(ROOT / "records/freeze.json")
assert fingerprints(ROOT / "submission") == freeze["source"]
assert fingerprints(args.adapter) == freeze["adapter"]
assert read(ROOT / "checkpoints/selected.json") == freeze["config"]
subprocess.run([sys.executable,"-X","utf8","-B",str(ROOT / "evaluate.py"),"test","--name",args.name,
    "--config",str(ROOT / "checkpoints/selected.json"),"--adapter",str(args.adapter.resolve()),
    "--official-package",str(args.official_package.resolve())],check=True)
output = REPO / "outputs/v13" / args.name
expected,actual = read(args.reference / "test.json"),read(output / "test.json")
assert expected["predictions"] == actual["predictions"], "SQL or real tool traces differ"
first_score,second_score = read(args.reference / "test-score.json"),read(output / "test-score.json")
assert first_score["per_question"] == second_score["per_question"]
fields = ("id","prompt","generated_text","sql","prompt_tokens","generated_tokens","kind")
expected_gen,actual_gen = lines(args.reference / "test-generations.jsonl"),lines(output / "test-generations.jsonl")
assert [{k:r.get(k) for k in fields} for r in expected_gen] == [{k:r.get(k) for k in fields} for r in actual_gen]
assert lines(args.reference / "test-memory.jsonl") == lines(output / "test-memory.jsonl")
ranking_fields = ("id","kind","prompt","candidates")
expected_rank,actual_rank = lines(args.reference / "test-ranking.jsonl"),lines(output / "test-ranking.jsonl")
assert [{k:r.get(k) for k in ranking_fields} for r in expected_rank] == [{k:r.get(k) for k in ranking_fields} for r in actual_rank]
metrics = read(output / "test-metrics.json")
assert metrics["within_cpu_model_limits"]
record = {"correct":second_score["correct"],"n":240,"sql_and_charged_traces_identical":True,
          "per_question_three_case_scores_identical":True,"model_prompts_text_tokens_identical":True,
          "schema_memory_events_identical":True,"semantic_scoring_prompts_candidates_identical":True,"prediction":metrics["prediction"],"score":metrics["score"]}
(output / "reproduction.json").write_text(json.dumps(record,indent=2),encoding="utf-8")
print(json.dumps(record),flush=True)
