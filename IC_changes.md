# I_C 融合策略改动记录

本文档汇总了为支持 **Internal Concat (I_C)** 及其变体 **I_C_SUM**（逐元素累加后注入）的改动，方便后续维护。每个条目给出文件、具体位置以及行为变化。

## 1. `llava/model/llava_arch.py`

1. **`LlavaMetaModel.initialize_vision_modules`**（约 60-150 行）：
   - 新增把 `model_args.layer_fusing_strategy` 写入 `self.config.layer_fusing_strategy`。
   - 当策略包含 `"I"` 时按 `layer_using_strategy` 构建对应数量的 `mm_projectors`；I_C 复用这套投影器用于 concat。
   - 保留 `mm_projector_f`，用于最终层特征或其他策略。

2. **`encode_images`**（约 180-235 行）：
   - 保持 E_M / E_D 的逻辑；
   - I_C 情况下不再返回 `image_features_list`，而是将每一层投影后的视觉 token 串接成单一块（在后续 `prepare_inputs_labels_for_multimodal` 中直接注入）。
   - I_C_SUM 新增逐元素求和：将各层 projector 输出堆叠后 `torch.sum`，再与 `mm_projector_f` 输出逐元素相加，最终只注入一段视觉 token，避免再次 concat。

3. **`prepare_inputs_labels_for_multimodal`**（约 240-455 行）：
   - 统一对所有策略生成 `new_input_embeds`、`image_token_mask`；
   - I_C 通过 concat 的方式直接把视觉 token 写入 `new_input_embeds`，并且不再返回 `image_features_list`（避免后续层重复注入）。

## 2. `llava/model/language_model/llava_llama.py`

1. **`LlavaLlamaForCausalLM.forward`**（约 120 行附近）：
   - `if images_features is None and image_features_list is not None` 分支新增 I_C / I_C_SUM 判断：
     - 非 I_C / I_C_SUM：`images_features = image_features_list + [image_features_f]`
     - I_C / I_C_SUM：置 `images_features = None`，表示视觉 token 已直接包含在 `inputs_embeds` 中。

## 3. `llava/model/language_model/modeling_llama.py`

1. **`LlamaDecoderLayer.__init__`**（约 720 行）：
   - 原逻辑只对 `"I" in layer_fusing_strategy` 设定 `has_cross`；现在增加：`self.layer_fusing_strategy = config.layer_fusing_strategy` 并对 I_C 直接 `self.has_cross = False`（防止进入层内注入分支）。

2. **`LlamaModel.__init__`**（约 1000 行）：
   - 存储 `self.layer_fusing_strategy = config.layer_fusing_strategy`，供 forward 判断。

3. **`LlamaModel.forward`**（约 1210-1260 行）：
   - 只在 I_D/I_M 的 `has_cross` 分支中读取 `images_features`；I_C 不再触发该分支。
   - 修复 I_M 判断为 `self.layer_fusing_strategy == 'I_M'`。

## 4. `llava/train/train.py`

1. **`ModelArguments.layer_fusing_strategy`**：
   - metadata 中注明可选值包含 `I_C` / `I_C_SUM`；在解析 CLI 时即可指定。

2. **模型加载阶段**：
   - LLaMA 分支在构建 `config` 后，若缺少 `layer_using_strategy/layer_fusing_strategy` 会从 `model_args` 写入。
   - `model.get_model().initialize_vision_modules(...)` 被调用以构建投影器。
   - 将 `training_args.layer_fusing_strategy = model_args.layer_fusing_strategy`，便于 Trainer 乃至 Dataset 获取策略。

3. **内部融合权重控制**：
   - 仅在 `"I" in model_args.layer_fusing_strategy` 时调用 `load_cross_attn_weights`、并解冻 `ucross` 模块；I_C 走 concat，不需要这些权重。

## 5. `scripts/MLVF/trainmini.sh`（可选脚本）

1. 增加了 `FUSING_STRATEGY` 变量，允许直接在脚本中切换到 `I_C`。脚本里分 pretrain/finetune 两部分调用 `train.py`，传入 `--layer_fusing_strategy ${FUSING_STRATEGY}`。

## 6. `.gitignore`

1. 新增忽略条目：
   - `checkpoint/`、`wandb/`、`*.log`、`__pycache__/`、`*.egg-info/`、`llava/.../deformable_attention/ops/build/` 等，确保训练产物不再被误提交。

---

> 总结：以上改动串联起 **Internal Concat** 流程——从 CLI 参数 → config → vision projector 初始化 → 数据准备 → Llama forward ——确保视觉 token 在输入阶段 concat，并避免后续层重复融合，达成 I_C 目标。

## 7. 追加说明（对话后续整理）

1. **External vs Internal**
   - 外部策略（E_D / E_M）在进入 LLM 之前就把全部视觉层融合成单一 embedding：
     - E_D：在 `encode_images()` 中按 `layer_using_strategy` 手动拼接/平均，再送入 `mm_projector_f`；
     - E_M：直接把多层特征输入到模块化 `mm_projector_f`（内部含 attention/MLP）里融合。
   - 内部策略（I_D / I_M / I_C）保留视觉 patch token 插入 `inputs_embeds`，在 LLM 层内继续处理。

2. **`image_features_list` vs `image_features_f`**
   - `image_features_list`：仅 I_D/I_M 使用，包含各个 projector 输出的视觉 token 列表，供层内注入；
   - `image_features_f`：视觉塔最后一层或全局特征经 `mm_projector_f` 得到的 embedding。传给模型时组合为 `image_features_list + [image_features_f]`；I_C 则直接设置 `images_features = None`。

3. **视觉 token 注入位置**
   - 在 `llava_arch.py::prepare_inputs_labels_for_multimodal()` 第 330~430 行，通过 `cur_new_input_embeds.append(cur_image_features)` 把视觉 token 逐段插入 text embedding，形成 `new_input_embeds`；这就是 LLM 第一层输入的视觉部分。

4. **内部策略差异**
   - I_D：在 `LlamaModel.forward()` 中（约 1220 行），第一次进入目标层时按 `image_token_mask` 将 `image_f` 直接加到 hidden state 上，所以 `decoder_layer(...)` 不需要 `image_feature` 参数。
   - I_M：同一处调用 `decoder_layer(..., image_feature=image_f, image_token_mask=...)`，在层内部的 `ucross_*` 模块使用视觉特征。
   - I_C：视觉 token 在输入阶段已 concat，`images_features` 设为 `None`，层内不再有额外注入。

5. **外部策略为什么不加载 cross 权重**
   - 因为 E_* 已经在 LLM 外部完成融合，不会实例化 `ucross` 模块，`load_cross_attn_weights` 只在 `"I" in layer_fusing_strategy` 时调用。

6. **训练脚本中的 Dataset 流程**
   - `train()` 在构建 `train_dataset` 时实例化 `SupervisedDataset`/`LazySupervisedDataset`（`llava/train/dataset.py`），它们在 `__getitem__` 中调用 `preprocess_multimodal` 和上述 `prepare_inputs_labels_for_multimodal`，完成 `<image>` 占位符替换及视觉特征拼接。

7. **I_C vs I_D**
   - I_C：多层视觉 token 直接 concat 成长序列，从第 0 层开始通过自注意力与文本互动；
   - I_D：保持视觉 token 占位符，待到指定层用 `image_f` 加法注入。

## 8. 新增 `18-23` 和 `23-23` 层选择策略（SigLIP I_C 支持）

为进一步探索不同层组合对 I_C 融合策略的影响，新增了两种层选择策略，专门针对 SigLIP 视觉编码器。

### 8.1 改动文件

#### `llava/model/llava_arch.py`

1. **`LlavaMetaModel.__init__`**（约 43-46 行）：
   ```python
   elif self.config.layer_using_strategy == '18-23':
       self.mm_projectors = nn.ModuleList([build_vision_projector(...) for _ in range(2)])
   elif self.config.layer_using_strategy == '23-23':
       self.mm_projectors = nn.ModuleList([build_vision_projector(...) for _ in range(1)])
   ```

2. **`LlavaMetaModel.initialize_vision_modules`**（约 107-110 行）：
   - 同样新增 `18-23` 和 `23-23` 分支，分别创建 2 个和 1 个 projector。

#### `llava/model/multimodal_encoder/siglip_encoder.py`

1. **`SigLipVisionTower.feature_select`**（约 583-597 行）：
   - 更新注释：`# For siglip, we tested 3-18-23, latter, 18-23, and 23-23`
   - 新增层索引映射：
     ```python
     if self.layer_using_strategy == '18-23':
         select_layer = [20, 25]    # layer 18, 23 + final layer for mm_projector_f
     if self.layer_using_strategy == '23-23':
         select_layer = [25, 25]        # layer 23 + final layer for mm_projector_f
     ```

### 8.2 策略说明

| 策略 | 选取的 SigLIP 层索引 | Projector 数量 | 用途 |
|------|---------------------|---------------|------|
| `18-23` | [20, 25] | 2 | 中间层 + 高层特征融合 |
| `23-23` | [25, 25] | 1 | 仅使用最高层特征（对照实验） |

> **注意**：SigLIP 共 27 层（删除最后 1 层后为 26 层），因此：
> - Layer 18（相对位置）→ 实际索引 20
> - Layer 23（相对位置）→ 实际索引 25
> - 最后一个索引始终用于 `mm_projector_f`

### 8.3 使用方式

在训练脚本中指定：
```bash
USING_STRATEGY="18-23"  # 或 "23-23"
FUSING_STRATEGY="I_C"

deepspeed llava/train/train.py \
    --layer_using_strategy ${USING_STRATEGY} \
    --layer_fusing_strategy ${FUSING_STRATEGY} \
    ...
```
