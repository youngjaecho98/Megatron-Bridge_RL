#!/usr/bin/env bash
set -uo pipefail

# --- cuda env ---
export CUDA_HOME=/usr/local/cuda
export CUDA_PATH=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export CPATH=$CUDA_HOME/targets/x86_64-linux/include:${CPATH:-}
export LIBRARY_PATH=$CUDA_HOME/targets/x86_64-linux/lib:${LIBRARY_PATH:-}
# --- end cuda env ---

# --- triton cuda tools ---
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
export TRITON_CUOBJDUMP_PATH=/usr/local/cuda/bin/cuobjdump
export TRITON_NVDISASM_PATH=/usr/local/cuda/bin/nvdisasm
# --- end triton cuda tools ---

export PYTHONPATH="/root/Megatron-Bridge/src:${PYTHONPATH:-}"

CKPT=${1:?Usage: bash inference.sh <checkpoint_path> <output_dir>}
OUTDIR=${2:?Usage: bash inference.sh <checkpoint_path> <output_dir>}

CUDA_DEVICE_MAX_CONNECTIONS=1 /opt/venv/bin/python -m torch.distributed.run --nproc-per-node=2 --master-port=29501 examples/models/vlm/nemotron_vl/finetune_nemotron_nano_v2_vl.py \
  --config-file /root/Megatron-Bridge/examples/models/vlm/nemotron_vl/conf/nemotron_nano_v2_vl_safewatch.yaml \
  --pretrained-checkpoint=/gpfs/public/artifacts/nemotron_nano_vl \
  train.skip_train=true \
  checkpoint.load="$CKPT" \
  checkpoint.load_optim=false \
  logger.eval_gt_jsonl_path=/gpfs/public/artifacts/SafeWatch-Bench-200K/sft_jsonl/sft_eval_v2.jsonl \
  logger.eval_dump_dir="$OUTDIR" \
  logger.eval_dump_max_records=null
