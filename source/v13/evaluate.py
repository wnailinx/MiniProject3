"""Unmodified official predictor/scorer; measured CPU reproduction for live models."""
import argparse
from collections import Counter
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
PACKAGE = REPO / "MP3_SQL_student_package"
EXPORT = ROOT / "submission"
from resource_usage import memory_tree


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def fingerprints(folder):
    return {p.relative_to(folder).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(folder.rglob("*")) if p.suffix in {".py", ".json", ".txt", ".safetensors"}}


def run(command, logfile, env):
    def limits():
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
    started = time.perf_counter()
    if sys.platform.startswith("linux"):
        import resource
        cpu_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    peak = 0
    with logfile.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(command, cwd=REPO, env=env, stdout=stream,
                                   stderr=subprocess.STDOUT,
                                   preexec_fn=limits if sys.platform.startswith("linux") else None)
        while process.poll() is None:
            if sys.platform.startswith("linux"):
                peak = max(peak, memory_tree(process.pid)[0])
            time.sleep(.1)
    if process.returncode:
        raise RuntimeError(logfile.read_text(encoding="utf-8"))
    cpu_seconds = None
    if sys.platform.startswith("linux"):
        cpu_after = resource.getrusage(resource.RUSAGE_CHILDREN)
        cpu_seconds = cpu_after.ru_utime+cpu_after.ru_stime-cpu_before.ru_utime-cpu_before.ru_stime
    return {"seconds": time.perf_counter()-started,"cpu_seconds":cpu_seconds, "peak_tree_rss_mib": peak/2**20,
            "command": command, "cpu_affinity": 4 if sys.platform.startswith("linux") else None}


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("split", choices=("train", "validation", "test"))
    cli.add_argument("--name", required=True)
    cli.add_argument("--config", type=Path, required=True)
    cli.add_argument("--adapter", type=Path)
    cli.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    cli.add_argument("--official-package",type=Path,default=PACKAGE)
    args = cli.parse_args()
    package = args.official_package.resolve()
    out = REPO / "outputs/v13" / args.name
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / f"{args.split}-score.json").exists(), "Use a fresh run name"
    config = read(args.config)
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    checkpoint = out / "config.json"
    before = {"agent":fingerprints(EXPORT), "official":fingerprints(package / "mp3_sql")}
    env = os.environ | {"PYTHONUTF8":"1", "PYTHONDONTWRITEBYTECODE":"1",
                        "OMP_NUM_THREADS":"4", "MKL_NUM_THREADS":"4", "OPENBLAS_NUM_THREADS":"1",
                        "TOKENIZERS_PARALLELISM":"false", "HF_HUB_OFFLINE":"1",
                        "MP3_MODEL_DEVICE":args.device,
                        "MP3_BASE_MODEL":os.environ.get("MP3_BASE_MODEL",str(REPO / "tmp/qwen3-0.6b")),
                        "MP3_GENERATIONS":str(out / f"{args.split}-generations.jsonl"),
                        "MP3_MEMORY_LOG":str(out / f"{args.split}-memory.jsonl"),
                        "MP3_RANK_LOG":str(out / f"{args.split}-ranking.jsonl")}
    if args.adapter:
        env["MP3_ADAPTER_DIR"] = str(args.adapter.resolve())
    for variable in ("MP3_V7_AUDIT", "MP3_V6_TELEMETRY"):
        env.pop(variable, None)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(EXPORT),str(package),env.get("PYTHONPATH"))))
    Path(env["MP3_GENERATIONS"]).write_text("",encoding="utf-8")
    Path(env["MP3_MEMORY_LOG"]).write_text("",encoding="utf-8")
    Path(env["MP3_RANK_LOG"]).write_text("",encoding="utf-8")
    prefix = [sys.executable,"-X","utf8","-B","-m"]
    prediction, scorefile = out / f"{args.split}.json", out / f"{args.split}-score.json"
    prediction_cost = run(prefix+["mp3_sql.predict","--split",args.split,"--workers","1","--method",
        str(EXPORT / "submission.py"),"--checkpoint",str(checkpoint),"--output",str(prediction)],
        out / f"{args.split}-predict.log",env)
    score_cost = run(prefix+["mp3_sql.score","--split",args.split,"--workers","4","--predictions",
        str(prediction),"--output",str(scorefile)],out / f"{args.split}-score.log",env)
    score, pred = read(scorefile), read(prediction)
    assert hashlib.sha256(prediction.read_bytes()).hexdigest() == score["predictions_sha256"]
    assert all(p["tool_calls"] <= 8 and len(p["trace"]) == p["tool_calls"]+1
               and p["trace"][0]["tool"] == "check" and p["trace"][-1]["tool"] == "submit"
               for p in pred["predictions"])
    assert before == {"agent":fingerprints(EXPORT),"official":fingerprints(package / "mp3_sql")}
    generations = [json.loads(s) for s in Path(env["MP3_GENERATIONS"]).read_text(encoding="utf-8").splitlines()]
    memory = [json.loads(s) for s in Path(env["MP3_MEMORY_LOG"]).read_text(encoding="utf-8").splitlines()]
    assert all(n <= 6 for n in Counter(e["id"] for e in generations).values())
    assert all(e["generated_tokens"] <= 2048 for e in generations)
    summary = {k:score[k] for k in ("split","n","correct","accuracy","mean_tool_calls")}
    summary |= {"device":args.device,"prediction":prediction_cost,"score":score_cost,
                "generation_calls":len(generations),"generated_tokens":sum(e["generated_tokens"] for e in generations),
                "source_hashes":before,"config":config,
                "adapter_hashes":fingerprints(args.adapter) if args.adapter else None}
    summary["memory_events"] = dict(Counter(e["event"] for e in memory))
    ranking = [json.loads(s) for s in Path(env["MP3_RANK_LOG"]).read_text(encoding="utf-8").splitlines()]
    summary["conditional_scoring"] = {"tasks":len(ranking),"candidates":sum(len(r["candidates"]) for r in ranking)}
    summary["runtime"] = {"python":sys.version,"sqlite":sqlite3.sqlite_version,
                          "packages":{k:version(k) for k in ("torch","transformers","peft","sqlglot","safetensors","accelerate")}}
    summary["within_cpu_model_limits"] = (args.device == "cpu" and prediction_cost["seconds"] <= 7200
        and score_cost["seconds"] <= 180 and prediction_cost["peak_tree_rss_mib"] <= 16*1024)
    (out / f"{args.split}-metrics.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps({k:summary[k] for k in ("split","correct","accuracy","generation_calls","generated_tokens")}),flush=True)


if __name__ == "__main__":
    main()
