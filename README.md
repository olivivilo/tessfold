# TessFold: Visual Token Compression for Document Understanding

This repository contains the current research implementation, configuration templates, pinned public-source metadata, and reproduction instructions. **Experimental results, pretrained models, compressor checkpoints, datasets, feature caches, and virtual environments are excluded.** Generated files are covered by `.gitignore`.

Two distinct backbone configurations are supported:

| Configuration | Frozen backbone | Compression-matrix insertion point | Generator |
|---|---|---|---|
| `und` | Official MonkeyOCRv2-B-Und, including its visual encoder, merger, and Qwen3-1.7B | Before the merger, or after the merger and before the LLM | Question-conditioned spatial tree / adaptive column partitioning |
| `native8b` | Complete native Qwen3-VL-8B-Instruct | After the native merger; the same matrix compresses the final visual features and all three DeepStack branches | Tree / Column / the separate StableColumn variant |

The native 8B configuration uses its own visual encoder and LLM; it is not an LLM replacement within MonkeyOCRv2-Und. In both configurations, the entire backbone is frozen and only the added compression-matrix generator is trained.

## Getting started

Run commands from the repository root. Use a Python 3.10 `venv` and a C++17 compiler (`g++`).

```bash
# CPU installation and regression tests that require no model weights
PYTHON_BIN=python3.10 DEVICE=cpu bash scripts/bootstrap_portable.sh
.venv/bin/python -m pytest
.venv/bin/python -m tessfold --help
```

For CUDA or Ascend, install compatible drivers and PyTorch wheels first; see [Installation](docs/INSTALL.md). Unit tests run on CPU, whereas full 8B training requires substantially more memory and compute.

## Models, data, and training

Copy a configuration from `configs/portable/` into the Git-ignored `configs/local/` directory. Edit the external model and data paths and the output directory.

```bash
mkdir -p configs/local
cp configs/portable/native_tree50.json configs/local/run.json
# After editing configs/local/run.json:
.venv/bin/python -m tessfold download-model --config configs/local/run.json
.venv/bin/python -m tessfold validate --config configs/local/run.json
.venv/bin/python -m tessfold pilot --config configs/local/run.json --device npu
.venv/bin/python -m tessfold train --config configs/local/run.json --device npu
.venv/bin/python -m tessfold evaluate --config configs/local/run.json --device npu
```

`download-model` explicitly downloads model files; the other commands load local models only. See [Data sources, format, and exact splits](docs/DATA.md) for data preparation. The repository contains public-source metadata and row-index reconstruction recipes, not question text or reference answers.

For distributed training or evaluation, use `torchrun`:

```bash
.venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=4 \
  -m tessfold train --config configs/local/run.json --device npu
# Resume with the same configuration, code, data, model, and world size:
.venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=4 \
  -m tessfold train --config configs/local/run.json --device npu --resume
```

Pilot runs also support `--resume`. By default, full training requires a successful pilot with the same configuration. Full training then starts from a fresh generator initialization rather than the pilot weights. Checkpoints are selected on the development set and locked before formal evaluation. Test scores never determine model selection.

## Repository structure

| Path | Purpose |
|---|---|
| `tessfold/` | Portable entry points for configuration, data validation, source pinning, training, checkpointing, and independent evaluation |
| `vqa/compression_generators_v2.py` | Original UND spatial generators, including Tree and Column |
| `vqa/tessfold_qwen3vl.py` | Frozen native 8B interface, Tree/Column, four-branch convex aggregation, and MRoPE handling |
| `vqa/tessfold_qwen3vl_stable.py` | Separate StableColumn variant with fixed detail-based partition allocation |
| `vqa/tessfold_native_partition.py` / `*.cpp` | C++ partition implementation that preserves reference arithmetic, with runtime compilation |
| `vqa/tessfold_*runtime.py`, `tessfold_prefix_cache.py` | Strict inference, segmented candidates, native partitioning, and independent-question KV management |
| `vqa/benchmark_metrics.py` | DocVQA, InfoVQA, ChartQA, OCRBench, DUE, WTQ, and DT-VQA metrics |
| `scripts/tessfold_accel/`, `tessfold_budget1075/`, `tessfold_repair/` | Original training, evaluation, timing, statistics, audit, and cluster orchestration scripts |
| Other modules in `scripts/` | Data conversion, compressor baselines and ablations, diagnostics, and report generation |
| `tests/` | CPU tests for mathematical properties, data isolation, training, and recovery without downloading a backbone |

See [Entry points](docs/ENTRYPOINTS.md) for compatibility details and [Method](docs/METHOD.md) for the architecture and its implementation.

## Reproduction scope

`vqa/` retains the current method implementations, with configurable model paths added. The portable `tessfold/` runner removes startup dependencies on historical result directories. Its default objective is full-vocabulary `KL(teacher || student)` under a shared teacher-generated response prefix, without gold-answer training or row entropy.

The portable runner uses a common development-set selection criterion, native task metrics, and exact gradient weighting for short final batches. It does not reproduce every historical protocol verbatim. For example, earlier UND experiments used fixed-epoch selection and field-level exact-match reporting. To reproduce a particular historical protocol, use the corresponding retained scripts and regenerate their required registrations. See [Reproducibility](docs/REPRODUCIBILITY.md) for these differences.

The source release has been checked with offline unit tests, a synthetic frozen-decoder training/resume/evaluation cycle, and reconstruction of public-manifest row indices. **This source preparation did not repeat full training with the real 8B or UND backbone. Synthetic tests do not establish reproduction of the paper's VQA scores.** Hardware, dependencies, dataset versions, and numerical kernels can affect results.

## Before publishing

```bash
git status --short
.venv/bin/python scripts/audit_source_repo.py
# Git is initialized locally. Replace this placeholder with your repository URL:
git remote add origin <YOUR_GITHUB_REPOSITORY_URL>
git commit -m "Add TessFold reproducible source"
git push -u origin main
```

For attribution to MetaCompress, MonkeyOCRv2, and related work, see [Third-party notices](THIRD_PARTY_NOTICES.md) and the [bibliography](docs/references.bib). The licensing status of the project's added code is documented in the notices; this source preparation does not assign a license on behalf of the rights holder.
