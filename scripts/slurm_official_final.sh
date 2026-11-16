#!/usr/bin/env bash
#SBATCH --job-name=monkey-und-final
#SBATCH --partition=a320m2tn910cu
#SBATCH --gres=npu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --output=artifacts/official-final-%j.log
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
UND_ADAPTER=$(.venv/bin/python -m scripts.select_official_und --checkpoint)
.venv-und/bin/python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 \
    -m scripts.eval_official_und --label finetuned_validation --mode validation --adapter "$UND_ADAPTER"
.venv/bin/python -m scripts.report_official_und --label finetuned_validation --mode validation --adapter "$UND_ADAPTER"
# Choose with validation BEFORE inspecting fine-tuned benchmark/test results.
.venv/bin/python -m scripts.select_official_und
.venv-und/bin/python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 \
    -m scripts.eval_official_und --label finetuned_full --adapter "$UND_ADAPTER"
.venv/bin/python -m scripts.report_official_und --label finetuned_full --adapter "$UND_ADAPTER"
.venv/bin/python -m scripts.report_official_training
