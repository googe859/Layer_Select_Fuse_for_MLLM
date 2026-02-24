#!/bin/bash
set -euo pipefail

# =============== Cache & Offline (推荐离线跑) ===============
CACHE="/public/home/h2023319033/.cache/huggingface"

export XDG_CACHE_HOME="$CACHE"
export HF_HOME="$CACHE"
export HF_HUB_CACHE="$CACHE/hub"
export HF_DATASETS_CACHE="$CACHE/datasets"
export TRANSFORMERS_CACHE="$CACHE/transformers"



# 离线开关（如果你确认缓存已预热齐全就开）
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# =============== Model(s) ===============
models=(
  "/public/home/h2023319033/opt/goyp/worksapce/PR/Layer_Select_Fuse_for_MLLM/checkpoint/past2/MobileLLaMA-1.4B-Base-I_C-finetune-3-20-25-siglip_14_665k"
  "/public/home/h2023319033/opt/goyp/worksapce/PR/Layer_Select_Fuse_for_MLLM/checkpoint/MobileLLaMA-1.4B-Base-baseline-finetune-clip_14_665k-hs23"
  "/public/home/h2023319033/opt/goyp/worksapce/PR/Layer_Select_Fuse_for_MLLM/checkpoint/MobileLLaMA-1.4B-Base-baseline-finetune-siglip_14_665k-hs20"
  "/public/home/h2023319033/opt/goyp/worksapce/PR/Layer_Select_Fuse_for_MLLM/checkpoint/MobileLLaMA-1.4B-Base-baseline-finetune-siglip_14_665k-hs25"
  "/public/home/h2023319033/opt/goyp/worksapce/PR/Layer_Select_Fuse_for_MLLM/checkpoint/MobileLLaMA-1.4B-Base-baseline-finetune-siglip_14_665k-hs26"
  "/public/home/h2023319033/opt/goyp/worksapce/PR/Layer_Select_Fuse_for_MLLM/checkpoint/MobileLLaMA-1.4B-Base-E_D-finetune-3-20-25-siglip_14_665k"
  "/public/home/h2023319033/opt/goyp/worksapce/PR/Layer_Select_Fuse_for_MLLM/checkpoint/MobileLLaMA-1.4B-Base-I_C-finetune-3-20-25-siglip_14_665k"
  "/public/home/h2023319033/opt/goyp/worksapce/PR/Layer_Select_Fuse_for_MLLM/checkpoint/past/MobileLLaMA-1.4B-Base-baseline-finetune-siglip_14_665k-hs20"
  "/public/home/h2023319033/opt/goyp/worksapce/PR/Layer_Select_Fuse_for_MLLM/checkpoint/past/MobileLLaMA-1.4B-Base-baseline-finetune-siglip_14_665k-hs22"
  "/public/home/h2023319033/opt/goyp/worksapce/PR/Layer_Select_Fuse_for_MLLM/checkpoint/past/MobileLLaMA-1.4B-Base-baseline-finetune-siglip_14_665k-hs25"
  "/public/home/h2023319033/opt/goyp/worksapce/PR/Layer_Select_Fuse_for_MLLM/checkpoint/past/MobileLLaMA-1.4B-Base-I_C-finetune-3-20-25-siglip_14_665k"
  "/public/home/h2023319033/opt/goyp/worksapce/PR/Layer_Select_Fuse_for_MLLM/checkpoint/past2/MobileLLaMA-1.4B-Base-baseline-finetune-siglip_14_665k-hs20"
  "/public/home/h2023319033/opt/goyp/worksapce/PR/Layer_Select_Fuse_for_MLLM/checkpoint/past2/MobileLLaMA-1.4B-Base-baseline-finetune-siglip_14_665k-hs22"
  "/public/home/h2023319033/opt/goyp/worksapce/PR/Layer_Select_Fuse_for_MLLM/checkpoint/past2/MobileLLaMA-1.4B-Base-baseline-finetune-siglip_14_665k-hs26"
  "/public/home/h2023319033/opt/goyp/worksapce/PR/Layer_Select_Fuse_for_MLLM/checkpoint/past2/MobileLLaMA-1.4B-Base-I_C-finetune-3-20-25-siglip_14_665k"
)


# =============== Tasks (一键全跑) ===============
# 你截图里这些 repo，对应 lmms_eval 常用 task 名大致如下：
tasks=(
   "pope"
  # "refcoco"
  #"gqa"
  # "seedbench"

 
)

# =============== Run ===============
for model_path in "${models[@]}"; do
  model_name=$(basename "$model_path")
  outdir="$model_path/eval"
  mkdir -p "$outdir/logs"

  echo "======================================================="
  echo "Model: $model_name"
  echo "Output dir: $outdir"
  echo "======================================================="

  for task in "${tasks[@]}"; do
    log_file="${outdir}/${model_name}_${task}.log"
    echo "[RUN] task=$task  ->  $log_file"

    CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes=4 -m lmms_eval \
      --model llava \
      --model_args pretrained="$model_path",conv_template=vicuna_v1,model_name="llava-mobile" \
      --tasks "$task" \
      --batch_size 1 \
      --log_samples \
      --log_samples_suffix "${model_name}_${task}" \
      --output_path "${outdir}/logs/" \
      > "$log_file" 2>&1

    echo "[DONE] task=$task"
  done
done

echo "All evaluations completed."
