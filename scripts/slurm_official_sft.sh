#!/usr/bin/env bash
#SBATCH --job-name=monkey-und-sft
#SBATCH --partition=a320m2tn910cu
#SBATCH --gres=npu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=artifacts/official-sft-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -u
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export TOKENIZERS_PARALLELISM=false
export HF_HOME="$PWD/artifacts/hf_cache"
export MODELSCOPE_CACHE="$PWD/artifacts/ms_cache"
export PYTHONPYCACHEPREFIX="$PWD/artifacts/python_cache/$SLURM_JOB_ID"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=4
export HCCL_CONNECT_TIMEOUT=1800
export HCCL_EXEC_TIMEOUT=7200
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export MAX_PIXELS=1003520
test -f models/MonkeyOCRv2-B-Und/VERIFIED
RESUME_ARGS=()
if [[ -n "${1:-}" ]]; then RESUME_ARGS=(--resume_from_checkpoint "$1"); fi
PROBE_ARGS=()
UND_OUTPUT_DIR="${UND_OUTPUT_DIR:-$PWD/artifacts/official_und/sft}"
if [[ -n "${UND_MAX_STEPS:-}" ]]; then
  PROBE_ARGS=(--max_steps "$UND_MAX_STEPS" --eval_steps 1 --save_steps 1)
fi
.venv-und/bin/python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 \
  -m swift.cli.sft \
  --model "$PWD/models/MonkeyOCRv2-B-Und" --model_type monkeyocrv2 --template monkeyocrv2_und \
  --external_plugins "$PWD/scripts/official_npu_plugin.py" --attn_impl sdpa \
  --dataset "$PWD/data/official_und/train.jsonl" --val_dataset "$PWD/data/official_und/validation.jsonl" \
  --split_dataset_ratio 0.0 --dataset_num_proc 1 --dataloader_num_workers 0 \
  --max_length 8196 --max_pixels 1003520 --truncation_strategy right \
  --packing False --padding_free False --use_logits_to_keep True \
  --train_type lora --lora_rank 8 --lora_alpha 32 --target_modules all-linear \
  --freeze_vit True --freeze_aligner False --freeze_llm False \
  --torch_dtype bfloat16 --gradient_checkpointing True \
  --gradient_checkpointing_kwargs '{"use_reentrant":false}' \
  --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --gradient_accumulation_steps 16 \
  --num_train_epochs 1 --learning_rate 1e-5 --warmup_ratio 0.05 --lr_scheduler_type cosine \
  --eval_on_start True --eval_strategy steps --eval_steps 50 --save_strategy steps --save_steps 50 --save_total_limit 3 \
  --logging_steps 1 --load_best_model_at_end True --metric_for_best_model loss --greater_is_better False \
  --report_to tensorboard --no_add_version --output_dir "$UND_OUTPUT_DIR" \
  "${RESUME_ARGS[@]}" "${PROBE_ARGS[@]}"
