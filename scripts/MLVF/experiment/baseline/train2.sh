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
FUSING_STRATEGY="E_D" # Options: E_D, E_M, I_D, I_M
USING_STRATEGY="3-20-25" # Options: 18, 3-18, 3-18-23, former, latter, all
MODEL_NAME="siglip_14_665k" # Specific model name, {Vsiual Encoder}_{LLM size}_{data size}

# Define paths
PRETRAIN_DATA_PATH="/home/share/llava1.5/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json"
PRETRAIN_IMAGE_FOLDER="/home/share/llava1.5/LLaVA-Pretrain/images"

MODEL_PATH="/home/share/llava1.5/model/MobileLLaMA-1.4B-Base"
VISION_TOWER="/home/share/llava1.5/model/siglip-so400m-patch14-384"

FINETUNE_DATA_PATH="/home/share/llava1.5/ft/llava_v1_5_mix665k_minitextvqa.json"
FINETUNE_IMAGE_FOLDER="/home/share/llava1.5/ft"


deepspeed_include_train="localhost:0,1,2,3,4,5,6,7"
per_device_train_bs=8
gradient_accumulation_steps_train=1
learning_rate_train=1e-3
weight_decay_train=5e-2
model_max_length_train=2048
warmup_steps_train=200
max_steps_train=-1
gradient_checkpointing_train=True

deepspeed_include_finetune="localhost:0,1,2,3,4,5,6,7"
per_device_finetune_bs=16
gradient_accumulation_steps_finetune=1
learning_rate_finetune=2e-5
weight_decay_finetune=0.
model_max_length_finetune=2048
warmup_ratio_finetune=0.03
max_steps_finetune=-1
gradient_checkpointing_finetune=True

pretrain_tag="bs${per_device_train_bs}-ga${gradient_accumulation_steps_train}-lr${learning_rate_train}-wd${weight_decay_train}-ml${model_max_length_train}-ws${warmup_steps_train}-ms${max_steps_train}-gc${gradient_checkpointing_train}-${deepspeed_include_train}"
finetune_tag="bs${per_device_finetune_bs}-ga${gradient_accumulation_steps_finetune}-lr${learning_rate_finetune}-wd${weight_decay_finetune}-ml${model_max_length_finetune}-wr${warmup_ratio_finetune}-ms${max_steps_finetune}-gc${gradient_checkpointing_finetune}-${deepspeed_include_finetune}"

PRETRAIN_OUT="./checkpoint/${BASE_MODEL_NAME}-${FUSING_STRATEGY}-pretrain-${USING_STRATEGY}-${MODEL_NAME}-${pretrain_tag}"
FINETUNE_OUT="./checkpoint/${BASE_MODEL_NAME}-${FUSING_STRATEGY}-finetune-${USING_STRATEGY}-${MODEL_NAME}-${finetune_tag}"

# Pretraining

deepspeed --include "${deepspeed_include_train}" "${TRAIN_SCRIPT}" \
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
    --per_device_train_batch_size "${per_device_train_bs}" \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps "${gradient_accumulation_steps_train}" \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --max_steps "${max_steps_train}" \
    --save_total_limit 4 \
    --learning_rate "${learning_rate_train}" \
    --weight_decay "${weight_decay_train}" \
    --warmup_steps "${warmup_steps_train}" \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length "${model_max_length_train}" \
    --gradient_checkpointing "${gradient_checkpointing_train}" \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --wandb_name "${BASE_MODEL_NAME}-${FUSING_STRATEGY}-pretrain-${USING_STRATEGY}-${MODEL_NAME}-${pretrain_tag}"

# Fine-tuning

deepspeed --include "${deepspeed_include_finetune}" "${TRAIN_SCRIPT}" \
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
    --per_device_train_batch_size "${per_device_finetune_bs}" \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps "${gradient_accumulation_steps_finetune}" \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --max_steps "${max_steps_finetune}" \
    --save_total_limit 5 \
    --learning_rate "${learning_rate_finetune}" \
    --weight_decay "${weight_decay_finetune}" \
    --warmup_ratio "${warmup_ratio_finetune}" \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length "${model_max_length_finetune}" \
    --gradient_checkpointing "${gradient_checkpointing_finetune}" \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --wandb_name "${BASE_MODEL_NAME}-${FUSING_STRATEGY}-finetune-${USING_STRATEGY}-${MODEL_NAME}-${finetune_tag}"
