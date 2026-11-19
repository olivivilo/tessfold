# Data Sources, Format, and Exact Splits

The repository contains no images, annotation text, gold answers, predictions, or dataset archives. `tessfold/sources/` stores only public-file download metadata, pinned revisions, checksums, and row-index recipes. Dataset licenses and image-source terms are determined by their respective providers.

## Preparing your own manifests

`train_rows`, `dev_rows`, and `eval_rows` each specify an external JSONL file. Each row follows this schema:

```json
{"id":"unique-question-id","document_id":"image-content-id","image":"/absolute/path/to/page.png","question":"Your question","answers":["Gold answer"],"task":"DocVQA"}
```

This is a schema illustration, not a distributed dataset sample. Supported tasks are `DocVQA`, `InfoVQA`, `ChartQA`, `OCRBench`, `DeepForm`, `KLC`, `WTQ`, and `DT-VQA`. Relative image paths resolve against `data_root`; other relative paths in a configuration resolve against the invocation directory.

Question IDs, document IDs, image paths, and actual image contents must be disjoint across the three splits. Multiple questions may share a document ID, but that ID must refer to the same image. `validate` checks these conditions; training repeats the checks and records SHA-256 values. The test set is never used to select hyperparameters or epochs.

## Public sources

| Task | Source |
|---|---|
| DocVQA, InfoVQA | DocVQA / InfographicVQA in `lmms-lab/DocVQA` |
| ChartQA | `ahmed-masry/ChartQA` |
| OCRBench | `echo840/OCRBench`; formal evaluation only |
| DeepForm, KLC, WTQ | Original DUE_Benchmark images and QA in `echo840/Monkey_Data` |
| DT-VQA | Original images and official QA in `shuozhang2/DT-VQA` |

Pinned revisions and file-level hashes are stored in `datasets.files.json`, `monkey_annotation.source.json`, and `dtvqa.source.json`. DT-VQA is hosted as a Hugging Face model repository, so its download URLs do not contain `/datasets/`.

## Reconstruction from public sources

Downloading is explicit and can require substantial time and disk space. Installation and tests do not trigger it. The original Monkey multipart ZIP is read on demand using HTTP Range; the server must support exact byte ranges.

```bash
.venv/bin/python -m pip install -r requirements-legacy.txt
.venv/bin/python -m tessfold download-data
# If your network requires a mirror:
# .venv/bin/python -m tessfold download-data --endpoint https://hf-mirror.com
.venv/bin/python -m scripts.prepare_full_vqa
.venv/bin/python -m scripts.append_due_vqa
.venv/bin/python -m scripts.append_dtvqa
```

The original preprocessing also runs `scripts.preflight_full_data` to remove empty supervision, check image bytes and duplicate decoded pixels, and record sequence lengths. This historical script expects the Qwen3-1.7B tokenizer specified by `configs/full.json` at `models/Qwen3-1.7B`. Its tokenizer files can be downloaded through the Hugging Face API:

```bash
.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-1.7B', local_dir='models/Qwen3-1.7B',
                  allow_patterns=['*.json', '*.txt', '*.model', '*.jinja'])
PY
.venv/bin/python -m scripts.preflight_full_data
.venv/bin/python -m tessfold prepare-cohorts
```

`prepare-cohorts` first verifies the pinned SHA-256 values of the converted `data/full/{train,val,test}.jsonl` files. It then reconstructs `data/train.jsonl` (20,000 QA), `data/dev.jsonl` (56 QA), and `data/eval.jsonl` (23,565 QA) by row index, and verifies each output's pinned SHA-256. These outputs are Git-ignored. The procedure does not resample data or silently replace a cohort to make a check pass.

If the same public-source conversion already exists elsewhere, use:

```bash
.venv/bin/python -m tessfold prepare-cohorts --source-root /path/to/converted/project
```

Set `data_root` to that source directory, or remap the image paths yourself. Changes in source versions, conversion order, Pillow decoding, or preprocessing filters can cause a source-hash mismatch. In that case, explicitly define new splits and report the protocol change rather than claiming exact reconstruction of the fixed cohorts.

The tokenizer download example above does not pin a revision and is used for preprocessing length statistics. Record the tokenizer version separately. The mandatory consistency checks for cohort reconstruction are the pinned source and final output SHA-256 values.
