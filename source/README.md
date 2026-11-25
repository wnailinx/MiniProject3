# Frozen v13 SQL repair submission

The complete official test result is 197/240 (82.08%). The evaluated agent uses one locally loaded Qwen/Qwen3-0.6B LoRA plus a 47-feature terminal ranker. The frozen inference tree is v13/submission/ and its configuration is v13/checkpoints/selected.json.

Paths in this README are relative to the source-export root: this directory (`source/`) in a repository clone, or the extracted root of the source ZIP. A repository name such as mp3-submit does not change these internal paths. Enter this directory before running the commands in REPRODUCE.md.

The finalized report PDF is v13/report.pdf. The report Markdown is deliberately excluded. This archive was generated after the report was explicitly marked final; the inference code, configuration and adapter remain frozen.

REPRODUCE.md contains the offline Linux commands. outputs/test.json and outputs/test-score.json are complete unchanged official artifacts. outputs also retains model prompts, token counts and conditional-scoring logs. evidence contains controlled ablations and the prior independent extraction result. The provided model adapter is a separate ZIP containing exactly two files; base model weights and benchmark data are excluded.

Only v13 inference code is deployed. v10/records/ contains three provenance/epoch-selection/base metadata files for the project's earlier adapter, not another runtime implementation. training-records/lora/ retains the original training source snapshots, configurations and logs as required for LoRA submissions. The v13 collection/label/fitting scripts retain original workspace input conventions for provenance; raw training trajectories/banks are not bundled. These training snapshots are not grading entry points and are not imported during inference. Instructors can reproduce inference with the supplied frozen adapter without retraining.

NOTICE preserves dataset/model/software attribution. Development assistance and other source acknowledgements belong in the report. Local packaging does not create a course submission issue.
