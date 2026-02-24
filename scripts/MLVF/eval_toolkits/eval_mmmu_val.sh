#!/bin/bash


# ================= 修复配置区域 =================

# 1. 修复权限问题：改用你的用户目录，不要用 /code

export HF_ENDPOINT=https://hf-mirror.com
export XDG_CACHE_HOME=/public/home/h2023319033/.cache/huggingface
export HF_HOME=/public/home/h2023319033/.cache/huggingface
export TRANSFORMERS_CACHE=/public/home/h2023319033/.cache/huggingface
export TRANSFORMERS_OFFLINE=1

# 2. 修复联网报错：注释掉这行，计算节点没网，跑这行必挂！
# conda install openjdk=8 -y

# 3. 修复路径问题：使用绝对路径，确保能找到模型
# 请确认这个路径下有 config.json 和 pytorch_model.bin
models=(
    "/public/home/h2023319033/opt/goyp/worksapce/PR/Layer_Select_Fuse_for_MLLM/checkpoint/past2/MobileLLaMA-1.4B-Base-I_C-finetune-3-20-25-siglip_14_665k"
)
tasks=mmmu_val



# ==============================================

for model_path in "${models[@]}"; do
    echo "Running evaluation for model: $model_path"
    model_name=$(basename "$model_path")

    outdir="/public/home/h2023319033/opt/goyp/worksapce/PR/Layer_Select_Fuse_for_MLLM/checkpoint/past2/${model_name}"
    mkdir -p "$outdir"

    log_file="${outdir}/${model_name}_${tasks}.log"


    echo "Starting evaluation for $model_name..."
    
    # 4. 移除 nohup，方便调试
    # 5. 加上 conv_template=vicuna_v1 防止 MobileLLaMA 报错
    CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes=4 -m lmms_eval \
        --model llava \
        --model_args pretrained="$model_path",conv_template=vicuna_v1,model_name="llava-mobile" \
        --tasks $tasks \
        --batch_size 1 \
        --log_samples \
        --log_samples_suffix $model_name \
        --output_path ./logs/ \
        > "$log_file" 2>&1

    echo "Evaluation completed. Log saved to ${model_name}_${tasks}.log"
done

echo "All evaluations completed."



