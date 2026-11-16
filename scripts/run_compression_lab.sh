#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
case "${1:-status}" in
 prepare) .venv/bin/python -m scripts.prepare_compression_lab ;;
 smoke) sbatch --job-name=lab-smoke scripts/slurm_compression_lab.sh smoke 1 ;;
 cache) sbatch --job-name=lab-cache --gres=npu:4 --cpus-per-task=32 --mem=128G scripts/slurm_compression_lab.sh cache 4 ;;
 baseline) sbatch --job-name=lab-baseline scripts/slurm_compression_lab.sh baseline 1 ;;
 screen)
   if [[ -n "$(squeue -h -u "$USER" --name=lab-screen -o %i)" ]]; then
     echo "An existing lab-screen job is active; retained without duplicate submission."
   else
     sbatch --job-name=lab-screen --gres=npu:8 --cpus-per-task=64 --mem=256G scripts/slurm_compression_lab.sh screen 8
   fi ;;
 resume) .venv/bin/python -m scripts.resume_compression_lab ;;
 advance) .venv/bin/python -m scripts.advance_compression_lab --submit ;;
 audit) sbatch scripts/slurm_compression_lab_audit.sh ;;
 plot) .venv/bin/python -m scripts.plot_compression_lab ;;
 report) .venv/bin/python -m scripts.report_compression_lab ;;
 status) squeue -u "$USER"; .venv/bin/python -m scripts.report_compression_lab ;;
 *) echo 'usage: bash scripts/run_compression_lab.sh prepare|smoke|cache|baseline|screen|resume|advance|audit|plot|report|status'; exit 2 ;;
esac
