# Reproduce the evaluated v13 agent

The PDF report describes the measured system. All commands below run from the source-export root: `source/` in a clone of the submission repository, or the extracted root of the source ZIP. The repository may be named `mp3-submit`; its name does not remove the internal `v13/` directory. The official starter package and the unchanged base model are provisioned separately; neither is distributed here. Extract the adapter ZIP to its own directory.

## 1. Environment

The measured environment is Linux, Python 3.12.3, SQLite 3.45.1, torch 2.7.1+cpu, Transformers 4.53.2, PEFT 0.17.1, Accelerate 1.10.1, Safetensors 0.8.0, SQLGlot 27.29.0, NumPy 2.5.5, huggingface-hub 0.36.2 and tokenizers 0.21.4. Python 3.11 or 3.12 is allowed by the project. SQLite ordering behavior can matter for under-specified ties, so use the measured environment for exact reproduction.

```bash
# Start at the repository root, or at the extracted source-ZIP root.
if [ -d source/v13/submission ]; then
  cd source
fi
test -f v13/submission/submission.py
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.7.1 \
  --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r v13/requirements-model.txt
export MP3_PROJECT="$PWD"
export MP3_OFFICIAL_PACKAGE=/absolute/path/to/MP3_SQL_student_package
export MP3_BASE_MODEL=/absolute/path/to/qwen3-base
export MP3_ADAPTER_DIR=/absolute/path/to/extracted-adapter
```

Before offline evaluation, download only the original approved base files if they are not already provisioned:

```bash
hf download Qwen/Qwen3-0.6B \
  --revision c1899de289a04d12100db370d81485cdf75e47ca \
  --include config.json generation_config.json model.safetensors \
            tokenizer.json tokenizer_config.json \
  --local-dir "$MP3_BASE_MODEL"
```

The two adapter files must match `v13/records/freeze.json`. Do not replace them with a newly trained checkpoint or merge them into the base. The repository/revision and original base/tokenizer hashes appear in `v10/records/base.json`.

## 2. Measured wrapper

The wrapper calls the unchanged official predictor/scorer, applies four-core affinity on Linux, writes full model logs, and measures time and process-tree RSS. Use fresh run names.

```bash
export CUDA_VISIBLE_DEVICES=""
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
python -X utf8 -B v13/evaluate.py validation \
  --name validation-recheck --config v13/checkpoints/selected.json \
  --adapter "$MP3_ADAPTER_DIR" --official-package "$MP3_OFFICIAL_PACKAGE"
python -X utf8 -B v13/evaluate.py test \
  --name test-recheck --config v13/checkpoints/selected.json \
  --adapter "$MP3_ADAPTER_DIR" --official-package "$MP3_OFFICIAL_PACKAGE"
```

Expected full results: validation **106/120**, test **197/240**. New files are under `outputs/v13/<run-name>/`.

## 3. Equivalent direct official commands

The unchanged original submission evidence is already included as `outputs/test.json` and `outputs/test-score.json`. To create fresh files at these exact paths, first use a fresh copy of the source archive or choose a different output directory so the original evidence remains available.

```bash
export PYTHONPATH="$MP3_PROJECT/v13/submission:$MP3_OFFICIAL_PACKAGE${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p outputs/direct-recheck
export MP3_MODEL_DEVICE=cpu
export MP3_GENERATIONS="$MP3_PROJECT/outputs/direct-recheck/test-generations.jsonl"
export MP3_RANK_LOG="$MP3_PROJECT/outputs/direct-recheck/test-ranking.jsonl"
python -X utf8 -B -m mp3_sql.predict --split test --workers 1 \
  --method v13/submission/submission.py \
  --checkpoint v13/checkpoints/selected.json \
  --output outputs/direct-recheck/test.json
python -X utf8 -B -m mp3_sql.score --split test --workers 4 \
  --predictions outputs/direct-recheck/test.json \
  --output outputs/direct-recheck/test-score.json
```

This direct form does not add the wrapper's CPU-affinity/resource measurement. Use the wrapper for the complete measured reproduction budget. Do not append repeated runs into old model log files.

## 4. Exact evidence comparison

```bash
python -X utf8 -B v13/reproduce.py \
  --official-package "$MP3_OFFICIAL_PACKAGE" \
  --adapter "$MP3_ADAPTER_DIR" --reference outputs \
  --name independent-recheck
```

This checks frozen inference identities before running, then compares all 240 SQL strings and actual charged traces, per-question three-instance scores, generation prompts/text/token counts and conditional-scoring prompts/candidates. The previously completed archive extraction evidence is `evidence/reproduction.json`.

## 5. Training records and source boundaries

`training-records/lora/full/` and `training-records/lora/holdout/` retain the actual LoRA training source, configurations and logs. `v10/records/` records training sources, base hashes and epoch selection. `v13/fit.py`, `collect.py`, `label.py` and `records/ranking-fit.json` record the ranker procedure. `evidence/additional-evidence.json` records ranking-source hashes and the actual NumPy 2.5.3 fitting environment. NumPy 2.5.5 above is the measured Linux inference environment, not a claim that ranking was fitted in that environment.

The v13 collection/label/fitting utilities and the original LoRA training-source snapshots are included for provenance. Only v13 inference code is deployed; v10/records contains base/data/epoch-selection metadata, not v10 runtime utilities. The snapshots preserve the original workspace conventions and do not promise standalone training without the original train inputs/trajectories. Raw training banks, databases, generated example files and environments are excluded. Training uses official train gold SQL and own train feedback; validation selects hyperparameters. Re-training may vary numerically across GPU environments; the supplied frozen adapter and selector are the grading assets.

## 6. Publication

Host the source/report ZIP and the separate two-file adapter ZIP at immutable HTTPS URLs. Select the complete original `outputs/test-score.json` on the course form, fill the student ID, prepare the submission, and actually create the generated GitHub issue. Local packaging alone does not submit a score. Keep artifacts unchanged and accessible through grading.
