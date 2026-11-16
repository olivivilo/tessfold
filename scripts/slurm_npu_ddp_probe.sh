#!/usr/bin/env bash
#SBATCH --job-name=monkey-ddp-probe
#SBATCH --partition=a320m2tn910cu
#SBATCH --gres=npu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --output=artifacts/npu-ddp-probe-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -u
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export TOKENIZERS_PARALLELISM=false
export PYTHONPYCACHEPREFIX="$PWD/artifacts/python_cache/$SLURM_JOB_ID"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=8
export HCCL_CONNECT_TIMEOUT=300
export HCCL_EXEC_TIMEOUT=600
.venv-npu/bin/python -m torch.distributed.run --standalone --nproc_per_node=2 \
    -m vqa.full_train --config configs/npu_ddp_probe.json --phase all
.venv/bin/python - <<'PY'
import json
from pathlib import Path
root=Path('artifacts/npu_ddp_probe_eager')
assert json.loads((root/'status.json').read_text())['state']=='complete'
assert json.loads((root/'execution.json').read_text())['world_size']==2
for s in (1,2):assert (root/f'stage{s}/complete.json').exists()
Path('artifacts/NPU_DDP_PROBE_PASSED').write_text('Two ranks: native vision cache, both training stages, checkpoint and evaluation passed\n')
PY
