#!/usr/bin/env bash
# Public entry: status | report | resume | train CONFIG.json | predict ARGS...
set -euo pipefail
cd "$(dirname "$0")/.."
case "${1:-status}" in
  status)
    squeue -u jchen524
    cat artifacts/meta_compress_strict/RUNNING_STATUS.md
    ;;
  report)
    exec .venv/bin/python -m scripts.report_meta_compress_strict
    ;;
  resume)
    exec sbatch scripts/slurm_meta_compress_strict_controller.sh
    ;;
  predict)
    shift
    exec sbatch scripts/slurm_meta_compress_strict_predict.sh "$@"
    ;;
  train)
    META_STRICT_CONFIG="${2:?Usage: train CONFIG.json}"
    META_STRICT_KIND=$(.venv/bin/python -c 'import json,sys;print(json.load(open(sys.argv[1]))["initialization"])' "$META_STRICT_CONFIG")
    case "$META_STRICT_KIND" in
      pca-whitened) META_STRICT_SCRIPT=scripts/slurm_meta_compress_strict_whitened.sh ;;
      pca-pool-null) META_STRICT_SCRIPT=scripts/slurm_meta_compress_strict_pool_null.sh ;;
      fourier-dense-position) META_STRICT_SCRIPT=scripts/slurm_meta_compress_strict_dense_position.sh ;;
      multiscale-position) META_STRICT_SCRIPT=scripts/slurm_meta_compress_strict_multiscale.sh ;;
      paper|fourier-position) META_STRICT_SCRIPT=scripts/slurm_meta_compress_strict.sh ;;
      *) exit 2 ;;
    esac
    META_STRICT_INSERTION=$(.venv/bin/python -c 'import json,sys;print(json.load(open(sys.argv[1]))["insertion"])' "$META_STRICT_CONFIG")
    if [[ "$META_STRICT_INSERTION" == post-norm || "$META_STRICT_INSERTION" == post-gelu ]]; then
      META_STRICT_SCRIPT=scripts/slurm_meta_compress_strict_inner.sh
    fi
    META_STRICT_DEVICES=$(.venv/bin/python -c 'import json,sys;print(json.load(open(sys.argv[1])).get("training_devices",2))' "$META_STRICT_CONFIG")
    if [[ "$META_STRICT_DEVICES" == 8 ]]; then
      META_STRICT_SCRIPT=scripts/slurm_meta_compress_strict_full.sh
    fi
    exec sbatch "$META_STRICT_SCRIPT" --config "$META_STRICT_CONFIG"
    ;;
  *) exit 2 ;;
esac
