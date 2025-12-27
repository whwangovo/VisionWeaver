#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_NAME="${CONFIG_NAME:-finetune_qwen_3b}"
cd "$ROOT_DIR"

export PYTHONPATH="${PYTHONPATH:-}:$ROOT_DIR"
export WANDB_PROJECT="hallucination"

NUM_GPUS=1
NUM_MACHINES=1
MACHINE_RANK=0
MAIN_PROCESS_IP="127.0.0.1"
MAIN_PROCESS_PORT=29500

ACCELERATE_ARGS=(
  --num_processes "$NUM_GPUS"
  --num_machines "$NUM_MACHINES"
  --machine_rank "$MACHINE_RANK"
  --main_process_ip "$MAIN_PROCESS_IP"
  --main_process_port "$MAIN_PROCESS_PORT"
)

accelerate launch "${ACCELERATE_ARGS[@]}" train_visionweaver.py \
  --config-name "$CONFIG_NAME"
