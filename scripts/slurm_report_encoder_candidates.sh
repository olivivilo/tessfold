#!/usr/bin/env bash
#SBATCH --job-name=encoder-proxy-report
#SBATCH --partition=a128m512u
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=artifacts/encoder-proxy-report-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
.venv/bin/python -m scripts.check_encoder_candidates
.venv/bin/python -m scripts.report_encoder_candidates --workers 16 --require-exact --require-line
if [[ ! -f artifacts/token_proxy/encoder_candidates_v1/subspace_audit.json ]]; then
 .venv/bin/python -m scripts.audit_encoder_subspace
fi
.venv/bin/python -m scripts.audit_encoder_candidate_outputs
.venv/bin/python -m scripts.sensitivity_encoder_candidate_statistics
.venv/bin/python -m scripts.write_encoder_candidate_report
