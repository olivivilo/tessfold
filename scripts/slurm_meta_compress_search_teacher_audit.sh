#!/usr/bin/env bash
#SBATCH --job-name=meta-teacher-audit
#SBATCH --partition=a128m512u
#SBATCH --cpus-per-task=32
#SBATCH --mem=100G
#SBATCH --time=01:00:00
#SBATCH --output=artifacts/meta-teacher-audit-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 TOKENIZERS_PARALLELISM=false OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8
export PYTHONPYCACHEPREFIX="/tmp/monkey_meta_pycache_${SLURM_JOB_ID}_${SLURM_NODEID:-0}"
AUDIT_PIDS=()
trap 'kill "${AUDIT_PIDS[@]}" 2>/dev/null || true' EXIT
for ((AUDIT_LOCAL=0; AUDIT_LOCAL<4; AUDIT_LOCAL++));do
  RANK="$AUDIT_LOCAL" WORLD_SIZE=4 .venv/bin/python -u -m scripts.audit_meta_compress_search_teacher &
  AUDIT_PIDS+=("$!")
done
AUDIT_FAILED=0
for AUDIT_PID in "${AUDIT_PIDS[@]}";do wait "$AUDIT_PID" || AUDIT_FAILED=1;done
exit "$AUDIT_FAILED"
