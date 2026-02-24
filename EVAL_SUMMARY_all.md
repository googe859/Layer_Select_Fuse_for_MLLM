# Eval Summary (All)

- Checkpoint root: H:\worksapce\PR\Layer_Select_Fuse_for_MLLM\checkpoint
- Includes: checkpoint/ and checkpoint/past*

|RelPath|Run|USING_STRATEGY|Fusing|Pretrain cfg|Finetune cfg|TextVQA(val) EM|MMMU(val) acc|POPE acc|POPE f1|POPE prec|POPE recall|POPE yes_ratio|GQA EM|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|checkpoint\MobileLLaMA-1.4B-Base-E_D-finetune-3-20-25-siglip_14_665k|MobileLLaMA-1.4B-Base-E_D-finetune-3-20-25-siglip_14_665k|3-20-25|E_D|bs=1, ga=8, gpus=4|bs=1, ga=16, gpus=4|0.4225+/-0.0067|0.2956|0.8574|0.8461|0.9190|0.7840|0.5000|0.5332+/-0.0044|
|checkpoint\MobileLLaMA-1.4B-Base-I_C-finetune-3-20-25-siglip_14_665k|MobileLLaMA-1.4B-Base-I_C-finetune-3-20-25-siglip_14_665k|3-20-25|I_C|bs=4, ga=4, gpus=4|bs=4, ga=8, gpus=4|0.4363+/-0.0068|0.3011|0.8596|0.8453|0.9409|0.7673|0.5000|0.5840+/-0.0044|
|checkpoint\past2\MobileLLaMA-1.4B-Base-I_C-finetune-3-20-25-siglip_14_665k|MobileLLaMA-1.4B-Base-I_C-finetune-3-20-25-siglip_14_665k|3-20-25|I_C|bs=?, ga=?, gpus=?|bs=1, ga=32, gpus=4|0.3709+/-0.0066|0.2756|0.8219|0.8205|0.8270|0.8140|0.5000|0.5122+/-0.0045|
|checkpoint\past\MobileLLaMA-1.4B-Base-I_C-finetune-3-20-25-siglip_14_665k|MobileLLaMA-1.4B-Base-I_C-finetune-3-20-25-siglip_14_665k|3-20-25|I_C|bs=?, ga=?, gpus=?|bs=2, ga=16, gpus=4|0.4407+/-0.0068|0.3067|0.8580|0.8427|0.9443|0.7609|0.5000|0.5929+/-0.0044|
|checkpoint\MobileLLaMA-1.4B-Base-baseline-finetune-clip_14_665k-hs23|MobileLLaMA-1.4B-Base-baseline-finetune-clip_14_665k-hs23|23|plain|bs=?, ga=?, gpus=?|bs=1, ga=16, gpus=4|0.3646+/-0.0066||0.8547|0.8451|0.9047|0.7929|0.5000|0.5377+/-0.0044|
|checkpoint\MobileLLaMA-1.4B-Base-baseline-finetune-siglip_14_665k-hs20|MobileLLaMA-1.4B-Base-baseline-finetune-siglip_14_665k-hs20|20|plain|bs=8, ga=2, gpus=4|bs=2, ga=16, gpus=4|0.3964+/-0.0067|0.2900|0.8734|0.8638|0.9348|0.8029|0.5000|0.5873+/-0.0044|
|checkpoint\past2\MobileLLaMA-1.4B-Base-baseline-finetune-siglip_14_665k-hs20|MobileLLaMA-1.4B-Base-baseline-finetune-siglip_14_665k-hs20|20|plain|bs=16, ga=1, gpus=4|bs=2, ga=16, gpus=4|0.3902+/-0.0067|0.2978|0.8658|0.8531|0.9422|0.7793|0.5000|0.5840+/-0.0044|
|checkpoint\past\MobileLLaMA-1.4B-Base-baseline-finetune-siglip_14_665k-hs20|MobileLLaMA-1.4B-Base-baseline-finetune-siglip_14_665k-hs20|20|plain|bs=?, ga=?, gpus=?|bs=4, ga=8, gpus=4|0.2509+/-0.0060|0.2933|0.8236|0.8192|0.8400|0.7993|0.5000|0.5169+/-0.0045|
|checkpoint\past2\MobileLLaMA-1.4B-Base-baseline-finetune-siglip_14_665k-hs22|MobileLLaMA-1.4B-Base-baseline-finetune-siglip_14_665k-hs22|22|plain|bs=16, ga=1, gpus=4|bs=2, ga=16, gpus=4|0.4201+/-0.0067|0.2900|0.8720|0.8610|0.9417|0.7931|0.5000|0.5923+/-0.0044|
|checkpoint\past\MobileLLaMA-1.4B-Base-baseline-finetune-siglip_14_665k-hs22|MobileLLaMA-1.4B-Base-baseline-finetune-siglip_14_665k-hs22|22|plain|bs=16, ga=1, gpus=4|bs=2, ga=16, gpus=4|0.4146+/-0.0067|0.2956|0.8613|0.8465|0.9477|0.7649|0.5000|0.5921+/-0.0044|
|checkpoint\MobileLLaMA-1.4B-Base-baseline-finetune-siglip_14_665k-hs25|MobileLLaMA-1.4B-Base-baseline-finetune-siglip_14_665k-hs25|25|plain|bs=16, ga=1, gpus=4|bs=2, ga=16, gpus=4|0.3524+/-0.0066|0.2978|0.8366|0.8265|0.8806|0.7787|0.5000|0.5393+/-0.0044|
|checkpoint\past\MobileLLaMA-1.4B-Base-baseline-finetune-siglip_14_665k-hs25|MobileLLaMA-1.4B-Base-baseline-finetune-siglip_14_665k-hs25|25|plain|bs=?, ga=?, gpus=?|bs=4, ga=8, gpus=4|0.3648+/-0.0066|0.2922|0.8544|0.8435|0.9119|0.7847|0.5000|0.5585+/-0.0044|
|checkpoint\MobileLLaMA-1.4B-Base-baseline-finetune-siglip_14_665k-hs26|MobileLLaMA-1.4B-Base-baseline-finetune-siglip_14_665k-hs26|26|plain|bs=?, ga=?, gpus=?|bs=2, ga=16, gpus=4|0.3481+/-0.0065|0.2989|0.8294|0.8165|0.8838|0.7587|0.5000|0.5350+/-0.0044|
|checkpoint\past2\MobileLLaMA-1.4B-Base-baseline-finetune-siglip_14_665k-hs26|MobileLLaMA-1.4B-Base-baseline-finetune-siglip_14_665k-hs26|26|plain|bs=?, ga=?, gpus=?|bs=2, ga=16, gpus=4|0.4374+/-0.0068|0.2978|0.8647|0.8506|0.9496|0.7702|0.5000|0.5915+/-0.0044|
