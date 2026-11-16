#!/usr/bin/env bash
set -euo pipefail
exec .venv/bin/python -u -m torch.distributed.run \
  --nnodes="$SLURM_NNODES" --nproc_per_node="$META_PROCESSES_PER_NODE" \
  --node_rank="$SLURM_NODEID" --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
  -m scripts.train_meta_compress --device cpu --threads "$META_THREADS" \
  --output "$META_OUTPUT" --max-items 20000 --epochs 2 --batch-size 36 --insertion post-merger
