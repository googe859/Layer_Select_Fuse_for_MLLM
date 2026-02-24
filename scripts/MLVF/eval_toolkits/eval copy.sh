#!/bin/bash

#!/bin/bash

# ================= 修复配置区域 =================



export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1


# 2. 修复联网报错：注释掉这行，计算节点没网，跑这行必挂！
# conda install openjdk=8 -y

# 3. 修复路径问题：使用绝对路径，确保能找到模型
# 请确认这个路径下有 config.json 和 pytorch_model.bin
models=(
    "/public/home/h2023319033/opt/goyp/worksapce/PR/Layer_Select_Fuse_for_MLLM/checkpoint/MobileLLaMA-1.4B-Base-baseline-finetune-siglip_14_665k-hs26"
)
tasks=textvqa_val
# ==============================================

for model_path in "${models[@]}"; do
    echo "Running evaluation for model: $model_path"
    model_name=$(basename $model_path)

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
        > "${model_name}.log" 2>&1

    echo "Evaluation completed. Log saved to ${model_name}.log"
done

echo "All evaluations completed."



