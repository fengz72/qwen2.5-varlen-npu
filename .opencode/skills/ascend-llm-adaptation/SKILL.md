---
name: ascend-llm-adaptation
description: >-
  适配 HuggingFace LLM 到华为 Ascend NPU 离线推理路径（PyTorch→AIR→OM→ACL）。
  全流程 7 步：环境校验、eager 跑通、算子融合、图信息决策、基础路径导出、
  编译期优化（NZ pass、限核）、profiling 驱动迭代。每个关键决策点（attention 类型、
  norm、位置编码、varlen、输出设计、lm_head）列出分支选项，以 Qwen2.5-0.5B 为
  已验证示例，可按模型类型扩展。已有 AIR 文件时可直接从 ATC 编译为 OM 开始。
---

# Ascend LLM 适配方法论

> 本文档只包含模型无关的适配流程和优化方向。具体代码实现、命令、踩坑细节
> 见 `reference/` 目录下各分类文件。

## 7 步框架

```
0. 环境校验          — CANN/torch_npu/transformers 版本 + NPU 可见性 + SoC 确认
1. Eager 跑通        — 加载模型 + 注册 NPU FIA + transformers eager 验证
2. 算子融合           — 发现可融合算子 + 搜索 NPU 融合算子 + monkey-patch 实施
3. 图信息决策         — varlen? 动态维度? frozen_parameter? ExportWrapper?
4. 跑通基础路径       — dynamo_export → ATC(默认) → OM → ACL 执行验证
5. 编译期优化         — NZ pass + MatMulV3、限核等，逐项验证收益
6. Profiling 迭代     — profiling → 回步骤2调整融合策略 → 重走4→5
```

**核心原则：** 先跑通最简路径（步骤4），建立可测量的 baseline，再逐项加优化（步骤5），最后用 profiling 数据驱动迭代（步骤6→步骤2）。每一步都可独立验证。

---

## 入口决策

根据当前进度选择起始步骤，无需从 Step 1 读起：

| 当前状态 | 起始步骤 |
|---------|---------|
| 只有模型名 / HuggingFace repo | Step 0 → Step 1 |
| eager 已跑通，未做算子融合 | Step 2 |
| 已有 patched model，准备导出 | Step 3 → Step 4 |
| 已有 AIR 文件 | Step 4（ATC 编译） |
| 已有 OM，性能不达标 | Step 6 → 回 Step 2 调整融合策略 |
| 已有 profiling 数据 | Step 6 分析 → 回 Step 2 |

---

## Step 0: 环境校验

开始适配前，确认环境就绪。版本不匹配是最常见的失败原因。

### 环境初始化

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

### 检查项

| 检查项 | 命令 | 说明 |
|--------|------|------|
| CANN 环境 | `echo $ASCEND_HOME_PATH` | 已 source set_env.sh，非空 |
| CANN 版本 | `cat $ASCEND_HOME_PATH/../version.cfg` | 记录版本号 |
| torch_npu | `python -c "import torch_npu; print(torch_npu.__version__)"` | 与 CANN 版本匹配 |
| torch | `python -c "import torch; print(torch.__version__)"` | 需支持 dynamo_export（≥ 2.1） |
| transformers | `python -c "import transformers; print(transformers.__version__)"` | 记录版本，见下方兼容性说明 |
| NPU 可见性 | `npu-smi info` | 至少 1 张卡 Available |
| SoC 型号 | `python -c "import torch_npu; print(torch_npu.npu.get_device_name())"` | 返回值直接用作 ATC `--soc_version`（如 `Ascend910_9382`） |

### 版本兼容性说明

**已验证版本组合：**

| 组合 | CANN | torch | torch_npu | transformers | SoC | 来源 |
|------|------|-------|-----------|-------------|-----|------|
| A | 9.0.0 | 2.9.0 | 2.9.0.post2 | 5.10.1 | Ascend910_9382 | Qwen2.5-0.5B |

> 后续验证新版本组合后，在此补充行。

**版本变化关注点：**
- **transformers API 变化**：attention forward 签名（`position_embeddings` vs
  `position_ids`）、`ALL_ATTENTION_FUNCTIONS` 注册接口等可能随大版本变化。
  适配新版本时需检查 modeling 源码确认签名。
- **SoC 型号**：`Ascend910_9382`（910C）与 910B 系列的 FIA 支持参数和可用
  核数不同，ATC `--soc_version` 必须与实际硬件一致。
- **torch_npu / torchair API 变化**：`dynamo_export`、`CompilerConfig`、
  `torchair.ops` 算子接口可能随 CANN 大版本变化。

### ✓ 完成判定

- [ ] 所有检查项通过，无报错
- [ ] `soc_version` 已确认，用于后续 ATC 命令
- [ ] transformers 版本已记录，用于确定 patch 签名

---

## Step 1: Eager 跑通

### 1.0 模型架构画像

加载模型前，先从 modeling 源码或 config.json 确认以下架构特征，决定后续
每个 Step 的分支选择：

| 检查项 | 常见选项 | 影响的 Step |
|--------|---------|------------|
| Norm 类型 | RMSNorm / LayerNorm | Step 2（融合算子选择） |
| 位置编码 | RoPE / 绝对位置 / ALiBi / 无 | Step 2（是否图外预计算） |
| Attention 方向 | causal / bidirectional | Step 1.2（FIA sparse_mode）、Step 3（mask） |
| Attention 分组 | MHA / GQA / MQA | Step 1.2（num_key_value_heads） |
| FFN 类型 | SwiGLU / GeLU / 其他 | Step 2（FFN 融合策略） |
| 输出 head | lm_head / custom head / 无 | Step 3.3（ExportWrapper 输出） |

> 已验证示例：Qwen2.5-0.5B = RMSNorm + RoPE(half) + causal GQA + SwiGLU + lm_head。
> 后续适配新模型时，先完成此画像，再按各 Step 的分支表选择对应路径。

### 1.1 加载模型（NPU 注意力）

用 `attn_implementation="npu_fia"` 加载模型到 NPU（fp16），让 transformers
从 `ALL_ATTENTION_FUNCTIONS` 注册表中查找名为 `"npu_fia"` 的注意力函数。

### 1.2 注册 NPU 推理注意力算子

将默认注意力替换为 `npu_fused_infer_attention_score`（推理算子，非训练用的
`npu_fusion_attention`），注册到 `ALL_ATTENTION_FUNCTIONS`。

**Attention 模式分支（由 Step 1.0 画像决定）：**

| 模式 | 适用场景 | sparse_mode | mask | 已验证 |
|------|---------|-------------|------|--------|
| causal | 文本生成、自回归编码 | 3 (rightDownCausal) | 上三角 bool | ✓ Qwen2.5 |
| bidirectional | 编码器(BERT)、embedding | 0 或不设 | 无需 mask | 待验证 |
| pair | 句对匹配 | 自定义 | block-diagonal | 待验证 |

> 后续遇到新的 attention 类型，在此补充分支。

**关键点：**
- 编译/eager 双路径：`torch.compiler.is_compiling()` 时用 `torchair.ops` 版本
  （为 dynamo_export 生成正确图节点），eager 时用 `torch_npu` 版本（支持
  golden 生成）。此模式对所有 `torchair.ops` 算子通用。
- GQA 通过 `num_key_value_heads` 原生支持，无需手动 `repeat_kv`。
- varlen 场景：`actual_seq_lengths` + `sparse_mode=3`（rightDownCausal）+
  固定 bool mask，让算子内部生成 block-diagonal 因果掩码。`actual_seq_lengths`
  和 `atten_mask` 在 Step 3 注入。
- 非 varlen 场景：可使用默认 attention，但推理算子对推理场景更优。

> 完整注册代码见 `reference/fia-registration.md`

### ✓ 完成判定

- [ ] 模型加载到 NPU（fp16），无 OOM
- [ ] `"npu_fia"` 已注册到 `ALL_ATTENTION_FUNCTIONS`
- [ ] eager forward 产出 logits，shape 正确
- [ ] golden logits 已保存（用于 Step 4 精度验证）

---

## Step 2: 算子融合

不同模型的算子结构不同，需要融合的算子也不一致。本步骤讲方法论：
**如何发现可融合的算子、如何找到对应的 NPU 融合算子、如何实施。**

### 2.1 发现可融合算子

1. **读 transformers 源码**：打开目标模型的 `modeling_<model>.py`，逐层
   检查 forward 中的算子链。重点关注多算子序列（Cast→Pow→Reduce→...）
   和重复计算（如 cos/sin 每层都算但结果不变）。
2. **跑一次 profiling**（Step 6 的方法提前用一次）：识别耗时 top-K 的
   算子和类型转换 kernel（如 fp32→fp16 Cast），这些是融合的首要目标。
3. **检查图内冗余**：某些计算每层重复但结果固定（如 RoPE 的 inv_freq
   @ position_ids），可提取到图外预计算。

### 2.2 搜索 NPU 融合算子

torch_npu 提供的融合算子在 `torch_npu` 命名空间下，可通过以下方式搜索：

| 搜索方式 | 方法 |
|---------|------|
| API 文档 | [hiascend torch_npu 文档](https://www.hiascend.com/document/detail/zh/Pytorch/) 查找 `npu_` 前缀的算子 |
| 源码搜索 | `grep -r "def npu_" torch_npu/` 或在安装目录中搜索 |
| 关键词联想 | 如看到 RMSNorm → 搜 `npu_rms_norm`；看到 rotary → 搜 `npu_apply_rotary` |
| torchair ops | 图编译专用版本在 `torchair.ops` 下（如 `npu_fused_infer_attention_score`） |

**注意：** API 文档只定义功能约束，**不做性能推荐**。某个融合算子是否
真能提速，必须通过 Step 6 profiling 验证。

### 2.3 实施融合（monkey-patch 模式）

通过 monkey-patch 替换 transformers 模块中的类/函数，**不修改 transformers
源码**。两种 patch 方式：

- **patch 类方法**：`<modeling_module>.<ModelClass>.forward = new_forward`
- **patch 模块函数**：`<modeling_module>.<function_name> = new_function`

> 占位符说明：`<modeling_module>` 是目标模型的 transformers modeling 模块
>（如 `modeling_qwen2`、`modeling_llama`），`<ModelClass>` / `<function_name>`
> 替换为目标模型对应的类/函数名。

### 2.4 图外预计算（融合的配套手段）

某些计算（如 RoPE cos/sin）每层重复且结果固定，可 patch forward 直接返回
预计算的缓存值。**一旦 patch 返回缓存，就必须在图外预计算并注入**，否则
图中没有该计算。

### 2.5 融合策略初始原则

| 原则 | 说明 |
|------|------|
| **先融合明确有收益的** | 如 RMSNorm（8→1 ops）、Attention（用推理算子替代训练算子） |
| **FFN/QKV 先不融合** | 融合算子的 tiling 可能不优，留到 Step 6 profiling 后决定 |
| **融合不一定更快** | tiling 妥协可能导致 MAC 利用率下降，拆分反而更优 |
| **始终 profile 验证** | 两种方案都跑，用 `aic_mac_ratio` 量化对比 |

**核心原则：** 融合策略不是一次定死的，Step 6 profiling 后可能推翻初始
选择，回到 Step 2 调整后重走 Step 4→5→6。

**算子融合分支（由 Step 1.0 画像决定）：**

| 算子类型 | 分支选项 | 已验证 |
|---------|---------|--------|
| Norm | RMSNorm→`npu_rms_norm` / LayerNorm→`npu_layer_norm` | ✓ RMSNorm |
| 位置编码 | RoPE→图外预计算+`npu_apply_rotary_pos_emb` / 绝对位置→保留图内 / 无→跳过 | ✓ RoPE |
| Attention | `npu_fused_infer_attention_score`（推理算子） | ✓ |
| FFN | SwiGLU→可试 `npu_ffn`(需profile验证) / 拆分MatMul / 其他 | ✓ 拆分 |
| QKV proj | 融合 / 不融合 | ✓ 不融合 |

> 后续遇到新的算子类型，在此补充分支。

> monkey-patch 模板、RoPE cos/sin 预计算代码、Qwen2.5-0.5b 融合实践示例
> 见 `reference/fusion-ops.md`

### ✓ 完成判定

- [ ] 已读 modeling 源码，识别出可融合算子链
- [ ] 已搜索到对应 NPU 融合算子（`npu_` 前缀 / `torchair.ops`）
- [ ] monkey-patch 已实施，eager 输出与未 patch 时一致
- [ ] 图外预计算已完成（如 RoPE cos/sin）
- [ ] 融合策略初始清单已记录（哪些融合 / 不融合 / 待 profiling 验证）

---

## Step 3: 图信息决策

本步骤做**影响图结构的决策**：哪些张量成为图输入、哪些维度动态、是否
需要 varlen/TND 转换。

### 3.1 序列拼接模式决策

| 模式 | 适用场景 | hidden_states | TND patch | ASL 注入 | 已验证 |
|------|---------|--------------|-----------|---------|--------|
| varlen | 变长序列批量推理 | 2D [T, D] | 需要 | 需要 | ✓ Qwen2.5 |
| 固定 batch | 等长序列批量 | 3D [B,S,D] 或 2D | 可选 | 不需要 | 待验证 |
| 单序列 | batch=1 推理 | 2D [S, D] | 不需要 | 不需要 | 待验证 |

> 后续遇到新的拼接模式，在此补充分支。

### Varlen 实施细节（选择 varlen 时）

**如果做 varlen**（变长序列批量推理）：

Varlen 将多条序列拼接为一个扁平 tensor `[T]`，其中 `T = sum(seq_lens)`。需要：

**a) 2D TND patch** — Monkey-patch `<ModelAttention>.forward`，让 hidden_states
全程保持 2D `[T, D]`。BSHD `[B, S, H]` 无法表达变长拼接，因此转 TND 是
**需求**，不是优化。同时，保持 2D 可消除 `nn.Linear` 对 3D+ 输入 reshape
产生的 GE Pack/Unpack 算子。

**规则：** 所有 `reshape` 必须使用 Python `int` 常量 + `-1`，不能提取
`SymInt`。这能防止 GE 生成动态 shape Pack 节点。

**b) token 拼接 + actual_seq_lengths 注入** — 构建累积序列长度 tensor，
注入到每层 attention，并禁用 transformers 自带的 `_update_causal_mask`。

**如果不做 varlen**（单条序列或固定 batch）：
- 跳过 TND patch、2D 转换、mask/ASL 注入
- batch=1：Pack 开销可忽略
- batch>1（BSHD）：profile 检查 Pack 开销；如需要则应用 2D patch

### 3.2 动态 shape 标记

标记哪些维度动态、哪些静态：

| 维度 | 动态？ | 原因 |
|------|--------|------|
| T（总 token 数） | 是（varlen） | 不同的 batch/seq 组合 |
| N（序列条数） | 是（varlen） | 序列数可变 |
| D（head_dim） | 否 | 模型结构固定 |
| cos/sin 维度 0 | 否 | 恒为 1 |
| cos/sin 维度 2 | 否 | 恒为 head_dim |

### 3.3 frozen_parameter + ExportWrapper 设计

启用 `frozen_parameter=1` 后，所有模型权重变为 Const 节点，图输入缩减为
仅用户可见的 tensor。

**ExportWrapper 输出设计分支（由业务场景决定）：**

| 输出模式 | 适用场景 | forward 返回 | 已验证 |
|---------|---------|-------------|--------|
| 末 token logits | 搜推打分、分类 | `lm_head(last_hidden)` | ✓ Qwen2.5 |
| 全 token logits | 生成式 prefill | `lm_head(all_hidden)` | 待验证 |
| embedding | 句向量、检索 | `norm(hidden)` + pooling | 待验证 |
| 自定义 head | 排序、回归 | `custom_head(hidden)` | 待验证 |
| KV cache | decode 阶段 | `hidden + kv_cache` | 待验证 |

> 后续遇到新的输出需求，在此补充分支。

**ExportWrapper** 还决定：
- 哪些 tensor 是图输入（Data 节点）、哪些是图常量（Const 节点）
- varlen 参数 + 缓存 cos/sin 如何在 forward 前注入

### 3.4 lm_head 处理决策

| 处理方式 | 适用场景 | 说明 | 已验证 |
|---------|---------|------|--------|
| 保留 | 文本生成、全词表输出 | 不修改 lm_head | ✓ Qwen2.5(默认) |
| 裁剪 | 搜推重排、限定候选集 | 裁剪到目标 token，减少 D2H 传输 | ✓ Qwen2.5(可选) |
| 替换 | 分类、回归 | 替换为 scoring head | 待验证 |
| 移除 | embedding 提取 | 不需要 lm_head | 待验证 |

> 后续遇到新的 head 处理需求，在此补充分支。

裁剪实现：如 `tie_word_embeddings=True`，先 clone 解绑再裁剪。

> TND patch 代码、varlen 拼接/注入函数、ExportWrapper 完整代码、
> 动态 shape 标记代码见 `reference/graph-decisions.md`

### ✓ 完成判定

- [ ] varlen 决策已做（是/否），如做 varlen 则 TND patch + ASL 注入已就绪
- [ ] 动态/静态维度已用 `mark_dynamic` / `mark_static` 标记
- [ ] ExportWrapper 已编写，forward 返回符合业务需求（全 token / 仅末 token）
- [ ] `frozen_parameter=1` 已确认
- [ ] lm_head 裁剪决策已做（如需要）

---

## Step 4: 跑通基础路径

**目标：** 在任何编译期优化之前，得到一个能正确输出的 OM，作为 Step 5 的
baseline。

> **重要：** NZ weight pass `.so` 一旦安装到 CANN vendor 目录，会对所有
> ATC 编译自动生效——没有开关。要获得真正的"无 NZ"baseline，确保此步骤
> 前未安装 pass，在 Step 5 再安装。

流程：
1. **dynamo_export**（PyTorch → AIR）：`frozen_parameter=1` + `dynamic=True`
2. **ATC 编译**（AIR → OM）：`--framework=1`，默认 `force_fp16`
3. **生成测试输入 + golden logits**：用于精度验证
4. **ACL 推理验证**：对比 OM 输出与 golden logits
5. **baseline benchmark**：记录 QPS、Execute avg、E2E avg、token 吞吐

已有 AIR 文件时，跳过 dynamo_export，直接从 ATC 编译开始。

> 完整命令和代码见 `reference/export-and-atc.md`

### ✓ 完成判定

- [ ] AIR 文件已生成（或已有）
- [ ] ATC 编译成功，OM 文件已生成
- [ ] ACL 推理输出与 golden logits 对比，精度达标
- [ ] baseline 指标已记录（QPS、Execute avg、E2E avg、token 吞吐）

---

## Step 5: 编译期优化

**目标：** 逐项应用编译期优化，每次与 Step 4 baseline 对比测量收益。

### 5.1 NZ Weight Pass + MatMulV3 替换

**原理：** Ascend Cube 单元以 FRACTAL_NZ 分块处理数据。ND 格式权重每次
访问产生运行时 TransData。编译期转换（常量折叠）可消除此开销。

**NZ pass 是跨模型复用的 GE graph pass**（C++ `.so`，注册到
`kAfterOriginGraphOptimize` 阶段）。它遍历图中所有 MatMul/MatMulV2 节点，
将 Const 权重从 ND 转为 FRACTAL_NZ，并替换为 MatMulV3。**安装一次即可，
适配新模型时无需修改 pass 代码。**

**适用条件：**
- 权重内轴（N）≥ 65536（足够大才值得 NZ 分块）
- 权重是 Const 节点（frozen_parameter=1）
- MatMul 是计算瓶颈（非访存瓶颈）

### 5.2 AICore 限核

用于多模型共享同一张 NPU 卡的场景，通过 `--aicore_num` 控制 AIC/AIV 核数。
需额外安装 GE Compiler 和 GE Executor。

### 5.3 其他编译选项

| 选项 | 标志 | 用途 |
|------|------|------|
| 调试日志 | `--log=debug` | 诊断 ATC 失败 |
| GE 图 dump | `DUMP_GRAPH_PATH=... DUMP_GE_GRAPH=2` | 检查图变换过程 |
| 精度模式 | （默认 force_fp16） | 对 fp16 模型已最优 |

> NZ pass 完整 C++ 源码、编译安装命令、限核命令见 `reference/compile-optimization.md`

### ✓ 完成判定

- [ ] NZ pass 安装后 ATC 重新编译，OM 中确认 MatMul→MatMulV3 替换生效
- [ ] 限核（如需要）已测试，最优 AIC/AIV 分配已确定
- [ ] 每项优化都与 Step 4 baseline 对比，收益数据已记录（QPS / Execute avg）

---

## Step 6: Profiling 迭代

**目标：** 用 profiling 数据定位瓶颈，然后**回到 Step 2** 调整融合策略，
重新导出（Step 4）、重新优化（Step 5）、重新测量。

### 分析流程

1. 按 `aclnn` 执行时间识别 top-K 算子
2. 检查 `aic_mac_ratio` — 低于 60% 说明 tiling 有问题
3. 检查 `aic_mte2_ratio` — 高说明访存瓶颈
4. 对可疑算子对比融合 vs 拆分方案
5. 在 Step 2 调整融合策略
6. 重新 export AIR + 编译 OM + benchmark 验证

### 关键指标

| 指标 | 良好范围 | 含义 |
|------|---------|------|
| `aic_mac_ratio` | >70% | Cube MAC 利用率 |
| `aic_mte2_ratio` | <80% | HBM 数据加载占比 |
| `cube_utilization` | >85% | Cube 整体忙碌率 |
| Execute avg | baseline | NPU 执行时间 |
| QPS | baseline | 每秒请求数 |
| Token 吞吐 | baseline | 每秒 token 数 |

> profiling 命令、Qwen2.5-0.5b 迭代示例和对比数据
> 见 `reference/profiling.md`

### ✓ 完成判定（单轮迭代）

- [ ] profiling 数据已解析，top-K 耗时算子已识别
- [ ] 瓶颈类型已定位（MAC 低 / mte2 高 / 其他）
- [ ] 融合策略已在 Step 2 调整
- [ ] 重新走 Step 4→5，新指标已与上轮对比
- [ ] 如仍有瓶颈 → 继续迭代；否则优化完成

### 停止条件

满足以下任一条件即可停止迭代：

- **收益收敛**：连续两轮迭代的 QPS 提升均 < 5%，继续优化性价比低
- **瓶颈转移**：主要瓶颈从计算（`aic_mac_ratio` 低）转为访存（`aic_mte2_ratio`
  高），融合策略不再有效，需从模型结构或精度层面调整
- **已达理论瓶颈**：`cube_utilization` > 85% 且 `aic_mac_ratio` > 70%

---

## 常见问题

> 所有踩坑记录（Pack/Unpack、eager 报错、Cast kernel、FFN 融合慢、
> 动态 shape 缺失、NZ pass 未生效等）见 `reference/pitfalls.md`
