#!/usr/bin/env bash
set -euo pipefail
export PYTHONPYCACHEPREFIX="/tmp/monkey_meta_pycache_${SLURM_JOB_ID}_${SLURM_NODEID:-0}"
exec .venv/bin/python -u -m torch.distributed.run \
  --nnodes="$SLURM_NNODES" --nproc_per_node="$META_SEARCH_PROCESSES" --node_rank="$SLURM_NODEID" \
  --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
  -m scripts.train_meta_compress_search --config "$META_SEARCH_FULL_CONFIG" --device cpu --threads "$META_SEARCH_THREADS"
