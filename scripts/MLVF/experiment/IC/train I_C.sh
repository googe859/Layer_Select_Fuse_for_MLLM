#!/bin/bash
set -euo pipefail

# Ensure we import the local repo `llava` package (avoid picking up an unrelated installed one).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null)"; then
  :
else
  REPO_ROOT="${SCRIPT_DIR}"
  while [[ "${REPO_ROOT}" != "/" && ! -f "${REPO_ROOT}/llava/train/train.py" ]]; do
    REPO_ROOT="$(dirname "${REPO_ROOT}")"
  done
  if [[ ! -f "${REPO_ROOT}/llava/train/train.py" ]]; then
    echo "Failed to locate repo root from ${SCRIPT_DIR}" >&2
    exit 1
  fi
fi
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
cd "${REPO_ROOT}"
TRAIN_SCRIPT="${REPO_ROOT}/llava/train/train.py"
DEEPSPEED_CONFIG="${REPO_ROOT}/scripts/zero2.json"
BASE_MODEL_NAME=MobileLLaMA-1.4B-Base

# Define common variables
FUSING_STRATEGY="I_C"  # Options: E_D, E_M, I_C, I_C_SUM, I_D, I_M

# Explicit vision hidden_states indices for BOTH CLIP and SigLIP (no implicit mapping).
# Examples:
#   - "20" -> hidden_states[20]
#   - "3-18-23" -> hidden_states[3], hidden_states[18], hidden_states[23]
#   - to emulate the old CLIP "3-18-23 also inject at 23" behavior, write "3-18-23-23"
USING_STRATEGY="${USING_STRATEGY:-3-18-23}"

MODEL_NAME="siglip_14_665k"  # {Visual Encoder}_{LLM size}_{data size}

# Define paths
PRETRAIN_DATA_PATH="/home/share/llava1.5/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json"
PRETRAIN_IMAGE_FOLDER="/home/share/llava1.5/LLaVA-Pretrain/images"

MODEL_PATH="/home/share/llava1.5/model/MobileLLaMA-1.4B-Base"
VISION_TOWER="/home/share/llava1.5/model/siglip-so400m-patch14-384"

FINETUNE_DATA_PATH="/home/share/llava1.5/ft/llava_v1_5_mix665k_minitextvqa.json"
FINETUNE_IMAGE_FOLDER="/home/share/llava1.5/ft"

export WANDB_MODE=offline

PRETRAIN_OUT="./checkpoint/${BASE_MODEL_NAME}-${FUSING_STRATEGY}-pretrain-${USING_STRATEGY}-${MODEL_NAME}"
FINETUNE_OUT="./checkpoint/${BASE_MODEL_NAME}-${FUSING_STRATEGY}-finetune-${USING_STRATEGY}-${MODEL_NAME}"

# Pretraining

deepspeed --include localhost:0,1,2,3,4,5,6,7 "${TRAIN_SCRIPT}" \
    --deepspeed "${DEEPSPEED_CONFIG}" \
    --model_name_or_path "${MODEL_PATH}" \
    --version plain \
    --data_path "${PRETRAIN_DATA_PATH}" \
    --image_folder "${PRETRAIN_IMAGE_FOLDER}" \
    --vision_tower "${VISION_TOWER}" \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --layer_using_strategy "${USING_STRATEGY}" \
    --layer_fusing_strategy "${FUSING_STRATEGY}" \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir "${PRETRAIN_OUT}" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --max_steps -1 \
    --save_total_limit 4 \
    --learning_rate 1e-3 \
    --weight_decay 5e-2 \
    --warmup_steps 200 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 3072 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --wandb_name "${BASE_MODEL_NAME}-${FUSING_STRATEGY}-pretrain-${USING_STRATEGY}-${MODEL_NAME}"

# Fine-tuning

deepspeed --include localhost:0,1,2,3,4,5,6,7 "${TRAIN_SCRIPT}" \
    --deepspeed "${DEEPSPEED_CONFIG}" \
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
    --layer_using_strategy "${USING_STRATEGY}" \
    --layer_fusing_strategy "${FUSING_STRATEGY}" \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir "${FINETUNE_OUT}" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --max_steps -1 \
    --save_total_limit 5 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 3072 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --wandb_name "${BASE_MODEL_NAME}-${FUSING_STRATEGY}-finetune-${USING_STRATEGY}-${MODEL_NAME}"
