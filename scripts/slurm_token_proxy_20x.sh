#!/usr/bin/env bash
#SBATCH --job-name=token-proxy-20x
#SBATCH --partition=a320m2tn910cu
#SBATCH --gres=npu:12
#SBATCH --cpus-per-task=96
#SBATCH --mem=384G
#SBATCH --time=08:00:00
#SBATCH --output=artifacts/token-proxy-20x-%j.log
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
# Each process independently runs the unchanged evaluator; no distributed model
# collectives. Two independent engines share each allocated logical NPU.
PROXY_DEVICES=12
PROXY_PROCESSES=24
PROXY_PIDS=()
trap 'kill "${PROXY_PIDS[@]}" 2>/dev/null || true' EXIT
for ((PROXY_RANK=0; PROXY_RANK<PROXY_PROCESSES; PROXY_RANK++)); do
    RANK="$PROXY_RANK" WORLD_SIZE="$PROXY_PROCESSES" LOCAL_RANK="$((PROXY_RANK % PROXY_DEVICES))" \
      .venv-und/bin/python -m scripts.eval_token_proxy --label random_v1_20x &
    PROXY_PIDS+=("$!")
done
PROXY_FAILED=0
for PROXY_PID in "${PROXY_PIDS[@]}"; do
    wait "$PROXY_PID" || PROXY_FAILED=1
done
exit "$PROXY_FAILED"
