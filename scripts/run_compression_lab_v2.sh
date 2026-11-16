#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
case "${1:-status}" in
 prepare) .venv/bin/python -m scripts.prepare_compression_lab_v2 ;;
 smoke) sbatch --job-name=v2-smoke scripts/slurm_compression_lab_v2.sh smoke 1 ;;
 screen)
  if [[ -n "$(squeue -h -u "$USER" --name=v2-screen -o %i)" ]]; then
   echo 'Existing v2-screen job retained.'
  else
   sbatch --job-name=v2-screen --gres=npu:14 --cpus-per-task=112 --mem=448G scripts/slurm_compression_lab_v2.sh screen 14
  fi ;;
 baseline) sbatch --job-name=v2-baselines --gres=npu:4 --cpus-per-task=32 --mem=128G scripts/slurm_compression_lab_v2.sh baseline 4 ;;
 advance) .venv/bin/python -m scripts.advance_compression_lab_v2 --submit ;;
 controller) sbatch scripts/slurm_compression_lab_v2_controller.sh ;;
 resume) .venv/bin/python -m scripts.resume_compression_lab_v2 ;;
 report) .venv/bin/python -m scripts.report_compression_lab_v2 --answers ;;
 final-report) .venv/bin/python -m scripts.report_compression_lab_v2_final ;;
 plot) .venv/bin/python -m scripts.plot_compression_lab_v2 ;;
 status) squeue -u "$USER"; .venv/bin/python -m scripts.report_compression_lab_v2 ;;
 *) echo 'usage: bash scripts/run_compression_lab_v2.sh prepare|smoke|screen|baseline|advance|controller|resume|report|final-report|plot|status'; exit 2 ;;
esac
