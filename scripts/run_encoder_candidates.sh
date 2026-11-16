#!/usr/bin/env bash
# Explicit entry point. Reuses immutable random_v1_20x X/P/answer caches.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=artifacts/token_proxy/encoder_candidates_v1
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
case "${1:-status}" in
 prepare)
  test -x .venv/bin/python
  test -f artifacts/token_proxy/random_v1_20x/audit.json
  if [[ ! -f "$OUT/protocol.json" ]]; then .venv/bin/python -m scripts.prepare_encoder_candidates; fi
  ;;
 submit)
  bash scripts/run_encoder_candidates.sh prepare
  # Reject duplicate submissions from this entry point while its jobs are active.
  if [[ -f "$OUT/jobs.txt" ]]; then
   while read -r PREVIOUS; do
    if [[ -n "$(squeue -h -j "$PREVIOUS" 2>/dev/null)" ]]; then
     echo "Existing pipeline job $PREVIOUS is active; use status instead."; exit 1
    fi
   done < "$OUT/jobs.txt"
  fi
  CACHE=$(sbatch --parsable scripts/slurm_encoder_probe.sh)
  if [[ -f "$OUT/evidence_head.pt" ]]; then
   HEAD="$CACHE"
  else
   HEAD=$(sbatch --parsable --dependency="afterok:$CACHE" scripts/slurm_train_encoder_evidence.sh)
  fi
  LOSS=$(sbatch --parsable --dependency="afterok:$HEAD" scripts/slurm_encoder_candidates.sh)
  if [[ -f "$OUT/line_head.pt" ]]; then
   LINE=$(sbatch --parsable --dependency="afterok:$HEAD" --partition=a128m512u --cpus-per-task=32 --mem=64G --time=02:00:00 --output=artifacts/encoder-line-replay-%j.log --wrap='export TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1; .venv/bin/python -m scripts.eval_encoder_relation --workers 32')
  else
   LINE=$(sbatch --parsable --dependency="afterok:$HEAD" scripts/slurm_encoder_relation.sh)
  fi
  OT=$(sbatch --parsable --dependency="afterok:$HEAD" scripts/slurm_exact_encoder_transport.sh)
  REPORT=$(sbatch --parsable --dependency="afterok:$LOSS:$LINE:$OT" scripts/slurm_report_encoder_candidates.sh)
  printf '%s\n' "$CACHE" "$HEAD" "$LOSS" "$LINE" "$OT" "$REPORT" > "$OUT/jobs.txt"
  cat "$OUT/jobs.txt"
  ;;
 status)
  squeue -u "$(id -un)"
  .venv/bin/python - <<'PY'
from pathlib import Path
p=Path('artifacts/token_proxy/encoder_candidates_v1')
for d in ['probe_features','losses','line_losses','exact_ot']:
 print(d,len(list((p/d).glob('*.pt' if d=='probe_features' else '*.json'))))
PY
  ;;
 check) .venv/bin/python -m scripts.check_encoder_candidates ;;
 report)
  .venv/bin/python -m scripts.report_encoder_candidates --workers 16 --require-exact --require-line
  if [[ ! -f "$OUT/subspace_audit.json" ]]; then .venv/bin/python -m scripts.audit_encoder_subspace; fi
  .venv/bin/python -m scripts.audit_encoder_candidate_outputs
  .venv/bin/python -m scripts.sensitivity_encoder_candidate_statistics
  .venv/bin/python -m scripts.write_encoder_candidate_report
  ;;
 *) echo "Usage: bash scripts/run_encoder_candidates.sh {prepare|submit|status|check|report}"; exit 2 ;;
esac
