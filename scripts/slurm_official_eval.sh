#!/usr/bin/env bash
#SBATCH --job-name=monkey-und-eval
#SBATCH --partition=a320m2tn910cu
#SBATCH --gres=npu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --output=artifacts/official-eval-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -u
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export TOKENIZERS_PARALLELISM=false
export HF_HOME="$PWD/artifacts/hf_cache"
export PYTHONPYCACHEPREFIX="$PWD/artifacts/python_cache/$SLURM_JOB_ID"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=4
test -f models/MonkeyOCRv2-B-Und/VERIFIED
.venv-und/bin/python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="${UND_EVAL_RANKS:-8}" \
    -m scripts.eval_official_und "$@"
.venv/bin/python -m scripts.report_official_und "$@"
