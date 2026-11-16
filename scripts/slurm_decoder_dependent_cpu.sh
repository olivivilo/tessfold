#!/usr/bin/env bash
#SBATCH --job-name=decoder-loss-cpu
#SBATCH --partition=a128m512u
#SBATCH --array=0-7
#SBATCH --cpus-per-task=128
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=artifacts/decoder_dependent_loss/cpu-%A_%a.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=${LOSS_CPU_THREADS:-8}
LOSS_PROCESSES=${LOSS_CPU_PROCESSES:-16}
LOSS_SHARDS=${LOSS_CPU_SHARDS:-8}
LOSS_PIDS=()
trap 'kill "${LOSS_PIDS[@]}" 2>/dev/null || true' EXIT
for ((LOSS_LOCAL=0; LOSS_LOCAL<LOSS_PROCESSES; LOSS_LOCAL++)); do
    RANK="$((SLURM_ARRAY_TASK_ID * LOSS_PROCESSES + LOSS_LOCAL))" \
    WORLD_SIZE="$((LOSS_SHARDS * LOSS_PROCESSES))" \
      .venv/bin/python -m scripts.eval_decoder_dependent --device cpu --batch-size 4 --work-queue "$@" &
    LOSS_PIDS+=("$!")
done
LOSS_FAILED=0
for LOSS_PID in "${LOSS_PIDS[@]}"; do wait "$LOSS_PID" || LOSS_FAILED=1; done
exit "$LOSS_FAILED"
