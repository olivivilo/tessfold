#!/usr/bin/env bash
#SBATCH --job-name=meta-benchmark-smoke
#SBATCH --partition=a128m512u
#SBATCH --cpus-per-task=32
#SBATCH --mem=100G
#SBATCH --time=01:00:00
#SBATCH --output=artifacts/meta-benchmark-smoke-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 TOKENIZERS_PARALLELISM=false OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8
export PYTHONPYCACHEPREFIX="/tmp/monkey_meta_pycache_${SLURM_JOB_ID}_${SLURM_NODEID:-0}"
SMOKE_PIDS=()
trap 'kill "${SMOKE_PIDS[@]}" 2>/dev/null || true' EXIT
for ((SMOKE_LOCAL=0; SMOKE_LOCAL<4; SMOKE_LOCAL++));do
  RANK="$SMOKE_LOCAL" WORLD_SIZE=4 .venv/bin/python -u -m scripts.benchmark_meta_compress_search \
    --config artifacts/meta_compress_search/configs/smoke.json \
    --checkpoint artifacts/meta_compress_search/runs/smoke/epoch1.pt --threads 8 --smoke &
  SMOKE_PIDS+=("$!")
done
SMOKE_FAILED=0
for SMOKE_PID in "${SMOKE_PIDS[@]}";do wait "$SMOKE_PID" || SMOKE_FAILED=1;done
exit "$SMOKE_FAILED"
