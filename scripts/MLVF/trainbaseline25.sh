#!/bin/bash
set -euo pipefail

# Ensure we import the local repo `llava` package (avoid picking up an unrelated installed one).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null)"; then
  :
else
  REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
cd "${REPO_ROOT}"
export MASTER_PORT="${MASTER_PORT:-29602}"

BASE_MODEL_NAME=MobileLLaMA-1.4B-Base
MODEL_NAME="siglip_14_665k"

# This repo interprets numeric `layer_using_strategy` as explicit vision `hidden_states` indices.
# Examples:
#   - "20" -> hidden_states[20]
#   - "3-18-23" -> hidden_states[3], hidden_states[18], hidden_states[23]
#   - duplicates allowed (e.g. "23-23")
# For SigLIP (siglip-so400m-patch14-384), the common last block output index is 25.
LAYER_USING_STRATEGY="${LAYER_USING_STRATEGY:-25}"

PRETRAIN_DATA_PATH="/publicssd/share/h13599/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json"
PRETRAIN_IMAGE_FOLDER="/publicssd/share/h13599/LLaVA-Pretrain/images"

MODEL_PATH="/publicssd/share/h13599/MobileLLaMA-1.4B-Base"
VISION_TOWER="/publicssd/share/h13599/siglip-so400m-patch14-384"

FINETUNE_DATA_PATH="/publicssd/share/h13599/llava_v1_5_mix665k/llava_v1_5_mix665k.json"
FINETUNE_IMAGE_FOLDER="/publicssd/share/h13599/llava_v1_5_mix665k"

PRETRAIN_OUT="./checkpoint/${BASE_MODEL_NAME}-baseline-pretrain-${MODEL_NAME}-hs${LAYER_USING_STRATEGY}"
FINETUNE_OUT="./checkpoint/${BASE_MODEL_NAME}-baseline-finetune-${MODEL_NAME}-hs${LAYER_USING_STRATEGY}"

# baseline LLaVA pretrain

# deepspeed --include localhost:0,1,2,4 llava/train/train.py \
#   --deepspeed ./scripts/zero2.json \
#   --model_name_or_path "${MODEL_PATH}" \
#   --version plain \
#   --data_path "${PRETRAIN_DATA_PATH}" \
#   --image_folder "${PRETRAIN_IMAGE_FOLDER}" \
#   --vision_tower "${VISION_TOWER}" \
#   --mm_projector_type mlp2x_gelu \
#   --tune_mm_mlp_adapter True \
#   --mm_vision_select_layer -2 \
#   --mm_use_im_start_end False \
#   --mm_use_im_patch_token False \
#   --layer_using_strategy "${LAYER_USING_STRATEGY}" \
#   --layer_fusing_strategy "plain" \
#   --bf16 True \
#   --output_dir "${PRETRAIN_OUT}" \
#   --num_train_epochs 1 \
#   --per_device_train_batch_size 16 \
#   --per_device_eval_batch_size 4 \
#   --gradient_accumulation_steps 1 \
#   --evaluation_strategy no \
#   --save_strategy steps \
#   --save_steps 500 \
#   --max_steps -1 \
#   --save_total_limit 4 \
#   --learning_rate 1e-3 \
#   --weight_decay 5e-2 \
#   --warmup_steps 200 \
#   --lr_scheduler_type "cosine" \
#   --logging_steps 1 \
#   --tf32 True \
#   --model_max_length 3072 \
#   --gradient_checkpointing False \
#   --dataloader_num_workers 4 \
#   --lazy_preprocess True \
#   --report_to wandb \
#   --wandb_name "${BASE_MODEL_NAME}-baseline-pretrain-${MODEL_NAME}-hs${LAYER_USING_STRATEGY}"

# # baseline LLaVA finetune

deepspeed --include localhost:0,1,2,4 llava/train/train.py \
  --deepspeed ./scripts/zero2.json \
  --model_name_or_path "${MODEL_PATH}" \
  --version v1 \
  --data_path "${FINETUNE_DATA_PATH}" \
  --image_folder "${FINETUNE_IMAGE_FOLDER}" \
  --vision_tower "${VISION_TOWER}" \
  --pretrain_mm_mlp_adapter "${PRETRAIN_OUT}/mm_projector.bin" \
  --mm_projector_type mlp2x_gelu \
  --mm_vision_select_layer -2 \
  --mm_use_im_start_end False \
  --mm_use_im_patch_token False \
  --layer_using_strategy "${LAYER_USING_STRATEGY}" \
  --layer_fusing_strategy "plain" \
  --image_aspect_ratio pad \
  --group_by_modality_length True \
  --bf16 True \
  --output_dir "${FINETUNE_OUT}" \
  --num_train_epochs 1 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --evaluation_strategy no \
  --save_strategy steps \
  --save_steps 1000 \
  --max_steps -1 \
  --save_total_limit 5 \
  --learning_rate 2e-5 \
  --weight_decay 0.0 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --tf32 True \
  --model_max_length 3072 \
  --gradient_checkpointing False \
  --dataloader_num_workers 4 \
  --lazy_preprocess True \
  --report_to wandb \
  --wandb_name "${BASE_MODEL_NAME}-baseline-finetune-${MODEL_NAME}-hs${LAYER_USING_STRATEGY}"
