# MiniProject3

DASE7506 Mini-project 3: feedback-guided SQLite repair with a locally loaded Qwen3-0.6B LoRA and calibrated candidate selection.

The included complete official scores are **197/240 (82.08%) on test** and **106/120 (88.33%) on validation**. These are local evaluation results; this repository does not claim instructor verification.

## Files

| Path | Contents |
| --- | --- |
| `source/` | Source export, frozen v13 agent, selected configuration, training records, model logs and reproduction evidence |
| `source.zip` | Packaged source export; its root corresponds to the contents of `source/` |
| `qwen3-adapter.zip` | Separate export containing exactly `adapter_config.json` and `adapter_model.safetensors` |
| `report.pdf` | Current report; identical to `source/v13/report.pdf` |
| `outputs/test.json` | Complete official predictions and charged traces |
| `outputs/test-score.json` | Complete unchanged official test score for the course form |

Base model weights, benchmark data, virtual environments and the editable report Markdown are not included.

## Reproduction

From a clone of this repository, enter `source/` and follow [REPRODUCE.md](source/REPRODUCE.md):

```bash
cd source
```

When using `source.zip`, start directly in its extracted root. The internal `v13/` directory is part of the export layout and is independent of the repository name. The official package and exact public base-model revision are provisioned separately. Extract the adapter ZIP to its own directory before evaluation. The measured reproduction uses offline Linux CPU inference.

## Course submission

The course form requires the complete `outputs/test-score.json`, an immutable HTTPS download link for `source.zip`, and a separate immutable HTTPS download link for `qwen3-adapter.zip`. The report is already inside the source ZIP. Uploading or pushing this repository does not create the course submission issue. Use the two prepared ZIPs rather than GitHub's automatically generated archive of the entire repository.

Attribution is retained in [NOTICE](source/NOTICE); method, training provenance, development assistance and evaluation details are in the report.
