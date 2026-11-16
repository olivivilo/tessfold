#!/usr/bin/env bash
#SBATCH --job-name=monkey-meta-eval
#SBATCH --partition=emergency_a320m2tn910c
#SBATCH --gres=npu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=artifacts/meta-compress-eval-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -u
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 TOKENIZERS_PARALLELISM=false
export HF_HOME="$PWD/artifacts/hf_cache"
.venv-und/bin/python -m scripts.eval_meta_compress --checkpoint artifacts/meta_compress/meta_compress.pt --output artifacts/meta_compress/eval --split val --limit-per-task 64
.venv-und/bin/python -m scripts.eval_meta_compress --checkpoint artifacts/meta_compress/meta_compress.pt --output artifacts/meta_compress/eval --split test --limit-per-task 64
