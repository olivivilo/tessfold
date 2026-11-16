#!/usr/bin/env bash
#SBATCH --job-name=monkey-vqa-full
#SBATCH --partition=a320m2tn910cu
#SBATCH --gres=npu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=6-00:00:00
#SBATCH --signal=B:USR1@900
#SBATCH --output=artifacts/full-npu-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -u
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export HF_HOME="$PWD/artifacts/hf_cache"
export TOKENIZERS_PARALLELISM=false
export PYTHONPYCACHEPREFIX="$PWD/artifacts/python_cache/$SLURM_JOB_ID"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=8
export HCCL_CONNECT_TIMEOUT=1800
export HCCL_EXEC_TIMEOUT=7200
test -f data/full/ALL_TASKS_READY
test -f artifacts/NPU_DDP_PROBE_PASSED
test -f artifacts/vision_cpu_npu_reference.json
.venv/bin/python - <<'PY'
import json
result=json.load(open('artifacts/vision_cpu_npu_reference.json'))
assert result['cosine_similarity'] >= .999 and result['relative_rmse'] <= .03
PY
CONFIG="${1:-configs/full.json}"
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 30000))
# Forward the scheduler's early warning to workers so optimizer checkpoints are saved.
CHILD_PID=''
trap 'if [[ -n "$CHILD_PID" ]]; then pkill -USR1 -P "$CHILD_PID" || true; fi' USR1 TERM
.venv-npu/bin/python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 \
    -m vqa.full_train --config "$CONFIG" --phase all &
CHILD_PID=$!
# A trapped warning interrupts bash's wait even while the workers are saving.
# Keep waiting for the actual torchrun exit so Slurm does not kill the checkpoint writer.
set +e
while true; do
    wait "$CHILD_PID"
    TRAIN_EXIT=$?
    if ! kill -0 "$CHILD_PID" 2>/dev/null; then break; fi
done
set -e
.venv/bin/python scripts/report_full.py
exit "$TRAIN_EXIT"
