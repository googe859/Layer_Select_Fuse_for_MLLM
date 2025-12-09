# I_C / I_C_SUM 融合策略改动记录

记录 `Layer_Select_Fuse_for_MLLM_pr` 中为支持 Internal Concat (**I_C**) 及其变体 **I_C_SUM**（多层 projector 输出逐元素求和再注入）而做的代码调整，便于 PR 维护。

## 1. `llava/model/llava_arch.py`

1. **`encode_images`**（约 180-235 行）  
   - I_C：仍把每层 projector 输出与 `mm_projector_f` 结果 concat 成单一块返回。  
   - I_C_SUM：先 `torch.stack` 多层 projector 输出，`torch.sum` 后与 `mm_projector_f` 输出逐元素相加，仅注入一段视觉 token，避免 concat 后的超长序列。

2. **`prepare_inputs_labels_for_multimodal`**（约 240-455 行）  
   - 不区分 I_C 与 I_C_SUM：视觉 token 在构造 `new_input_embeds` 时已经注入，因此返回的 `image_features_list` 只保留给 I_D / I_M 使用。

## 2. `llava/model/language_model/llava_llama.py`

- `forward()` 在 `images_features` 为空且 `image_features_list` 存在时，若策略为 I_C 或 I_C_SUM 就直接置 `images_features = None`，避免重复注入视觉 token；其他策略仍把 `image_features_list + [image_features_f]` 交给模型。

## 3. `llava/model/language_model/modeling_llama.py`

- `LlamaDecoderLayer.__init__` 检查 `layer_fusing_strategy`，对 I_C 与 I_C_SUM 均设置 `self.has_cross = False`，避免错误地进入 cross 模块。  
- 其余需要 `== "I_C"` 的判断同样扩展为 `in ("I_C", "I_C_SUM")`。

## 4. `llava/train/train.py`

- `ModelArguments.layer_fusing_strategy` 的 metadata 增加 `I_C_SUM`，CLI 解析时即可传入该值。  
- `training_args.layer_fusing_strategy` 和 `model.config.layer_fusing_strategy` 会继承这个新枚举，确保下游模块能识别。

## 5. 使用提示

- 训练 / 推理脚本中可通过 `--layer_fusing_strategy I_C_SUM` 开启新策略。  
- I_C_SUM 仍然需要与 I_C 相同的 projector 初始化（按 `layer_using_strategy` 个数创建），只是注入前把多层视觉 token 按元素求和，因此推荐搭配更大的 `model_max_length` 以容纳视觉+文本序列。
