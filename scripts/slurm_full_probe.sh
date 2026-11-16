#!/usr/bin/env bash
#SBATCH --job-name=monkey-full-probe
#SBATCH --partition=a128m512u
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --output=artifacts/full-probe-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export OPENBLAS_NUM_THREADS=1
export HF_HOME="$PWD/artifacts/hf_cache"
.venv/bin/python -m vqa.full_train --config configs/full_probe.json --phase all
# Replay from saved optimizer/LM state; complete stages must not retrain.
.venv/bin/python -m vqa.full_train --config configs/full_probe.json --phase train
