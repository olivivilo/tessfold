#!/usr/bin/env bash
#SBATCH --job-name=decoder-dependent-loss
#SBATCH --partition=a320m2tn910cu
#SBATCH --gres=npu:12
#SBATCH --cpus-per-task=96
#SBATCH --mem=384G
#SBATCH --time=08:00:00
#SBATCH --output=artifacts/decoder_dependent_loss/run-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -u
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 TOKENIZERS_PARALLELISM=false
export HF_HOME="$PWD/artifacts/hf_cache"
export PYTHONPYCACHEPREFIX="$PWD/artifacts/python_cache/$SLURM_JOB_ID"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
LOSS_DEVICES=${LOSS_DEVICES:-12}
LOSS_PROCESSES=${LOSS_PROCESSES:-24}
LOSS_PIDS=()
trap 'kill "${LOSS_PIDS[@]}" 2>/dev/null || true' EXIT
for ((LOSS_RANK=0; LOSS_RANK<LOSS_PROCESSES; LOSS_RANK++)); do
    RANK="$LOSS_RANK" WORLD_SIZE="$LOSS_PROCESSES" LOCAL_RANK="$((LOSS_RANK % LOSS_DEVICES))" \
      .venv-und/bin/python -m scripts.eval_decoder_dependent "$@" &
    LOSS_PIDS+=("$!")
done
LOSS_FAILED=0
for LOSS_PID in "${LOSS_PIDS[@]}"; do wait "$LOSS_PID" || LOSS_FAILED=1; done
exit "$LOSS_FAILED"
