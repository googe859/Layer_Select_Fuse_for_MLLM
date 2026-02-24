# 1) 固定到你截图里的缓存根目录（建议保持一致）
export HF_HOME=/public/home/h2023319033/.cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TRANSFORMERS_CACHE=$HF_HOME/transformers

# 2) 关闭离线（预热时必须）
unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE TRANSFORMERS_OFFLINE
unset HF_ENDPOINT
export HF_ENDPOINT=https://hf-mirror.com
# 3) 预热这些数据集（生成 modules 缓存）
python - <<'PY'
import datasets

names = [
#   "lmms-lab/POPE",
#   "AI4Math/MathVista",
#   "lmms-lab/ai2d",
#   "lmms-lab/ChartQA",
#   "lmms-lab/ScienceQA",
#   "lmms-lab/DocVQA",
#   "lmms-lab/mmbench",
#   "lmms-lab/textvqa",
#   "lmms-lab/MMMU",

#   "lmms-lab/MME",
  "lmms-lab/GQA",
#   "lmms-lab/SEED-Bench",
#   "echo840/OCRBench",
#   "nyu-visionx/CV-Bench",
#   "lmms-lab/RealWorldQA",
#   "lmms-lab/RefCOCO",
]

for name in names:
    print(f"=== caching: {name} ===")
    try:
        datasets.load_dataset(name,"testdev_balanced_instructions")
        print("OK\n")
    except Exception as e:
        print("FAILED:", repr(e), "\n")
PY
