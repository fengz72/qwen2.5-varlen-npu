# Qwen2.5 Varlen NPU 推理优化总结

## 1. 项目概述

基于 Huawei Ascend NPU 的 Qwen2.5-0.5B 变长 (varlen) 推理框架，通过 CANN 融合算子和 TorchAir 图编译实现高性能推理。

### 环境信息

| 项目 | 版本 |
|------|------|
| 模型 | Qwen2.5-0.5B (24层, hidden=896, intermediate=4864, vocab=151936) |
| 硬件 | Ascend910 (A2), 64GB HBM |
| CANN | 9.0.0 |
| PyTorch | 2.x + torch_npu |
| torchair | torch_npu.dynamo.torchair |

### 核心设计

1. **Varlen 打包**: 多条不同长度文本拼接为一条序列, 一次推理处理多个请求
2. **NPU 推理算子**: `npu_fused_infer_attention_score` 替代手动 attention 计算
3. **图编译**: `torchair` GE 图模式 + `dynamic=True`, 一次编译多 shape 复用
4. **融合算子**: Monkey-patch 替换 RMSNorm/RoPE/FFN 为 NPU 融合算子

---

## 2. 模块结构与代码实现

```
qwen_varlen/
├── __init__.py       # 包入口, 导出公共接口
├── attention.py      # 自定义 NPU 推理 attention (npu_fused_infer_attention_score)
├── varlen_utils.py   # 变长输入处理 + RoPE cos/sin 图外预计算
├── fusion_ops.py     # 融合算子替换 (RMSNorm/RoPE/FFN)
├── profiling.py      # Profiling 采集封装
└── run_infer.py      # 主入口 (CLI)
```

### 2.1 attention.py — 自定义 Attention

**功能**: 通过 `ALL_ATTENTION_FUNCTIONS.register` 注入自定义 attention, 不修改 transformers 源码。

**核心逻辑**:
- 接收 Qwen2Attention.forward 的 BNSD [B, N, S, D] 格式 q/k/v
- 转换为 TND [S, N, D] (varlen packed)
- 调用 `npu_fused_infer_attention_score`:
  - `input_layout="TND"`: varlen 模式
  - `actual_seq_lengths`: 以 tensor 传入, 避免 dynamo specialize
  - `sparse_mode=3`: rightDownCausal, 算子内部生成 block-diagonal causal mask
  - `num_key_value_heads`: 原生支持 GQA, 无需手动 repeat_kv
- 返回 BSND [1, S, N, D]

**关键代码**:
```python
ALL_ATTENTION_FUNCTIONS.register("npu_fia", npu_fia_varlen_forward)
# 加载模型时: attn_implementation="npu_fia"
```

### 2.2 varlen_utils.py — 变长输入处理

**功能**: 将多条文本拼接为 varlen 格式, 预计算 attention 参数并注入模型。

**三个函数**:

1. `prepare_varlen_inputs(tokenizer, input_texts)`:
   - 将多条文本 token 化后拼接为 [1, total_len]
   - 生成 position_ids [1, total_len] (每条文本从 0 开始)
   - 计算 seq_lens 和 cum_seq_lens (累积长度)

2. `precompute_rope_cos_sin(model, total_len, device)`:
   - 图外预计算 RoPE 的 cos/sin, 直接生成 fp16
   - 避免图内 Cast(fp32→fp16) kernel (profiling 显示该 Cast 耗时 206us)
   - 直接内联计算, 不依赖 rotary_emb.forward (已被 monkey-patch)
   - 结果注入 `rotary_emb._cached_cos` / `_cached_sin`

3. `setup_varlen_attention(model, cum_seq_lens, device)`:
   - 构建固定 2048x2048 bool causal mask (用于 sparse_mode=3)
   - 将 `actual_seq_lengths_tensor` 和 `atten_mask` 注入每层 self_attn
   - 禁用 transformers 自带 `_update_causal_mask` (由推理算子内部处理)
   - 调用 `precompute_rope_cos_sin` 预计算并注入 cos/sin

### 2.3 fusion_ops.py — 融合算子替换

**功能**: 通过 monkey-patch 将 Qwen2 小算子替换为 NPU 融合算子。

**4 个替换项**:

| # | 替换目标 | 原始小算子 | 融合算子 | 注入方式 |
|---|---------|-----------|---------|---------|
| 1 | RMSNorm | Cast→Pow→ReduceMean→Add→Rsqrt→Mul→Cast→Mul (8个) | `npu_rms_norm` (1个) | `Qwen2RMSNorm.forward =` |
| 2 | RoPE rotate | rotate_half(StridedSlice+Neg+Cat)+Mul+Add (12个) | `npu_rotary_mul` (2个) | `modeling_qwen2.apply_rotary_pos_emb =` |
| 3 | RoPE cos/sin | 图内 Cast(206us)+MatMul+Cos+Sin (5个) | 图外预计算注入 (0个图内kernel) | `Qwen2RotaryEmbedding.forward =` |
| 4 | FFN/MLP | 2×MatMul+Cat+SwiGLU+MatMul (5个) | `npu_ffn(swiglu)` (1个) | `Qwen2MLP.forward =` |

**npu_ffn 权重拼接细节**:
```
nn.Linear weight: [out_features, in_features]
  gate_proj.weight: [4864, 896]
  up_proj.weight:   [4864, 896]
  down_proj.weight: [896, 4864]

npu_ffn 要求:
  weight1: [K, N] = [896, 9728] = cat([up_w.T, gate_w.T], dim=1)
  weight2: [K, N] = [4864, 896] = down_w.T

验证确认: npu_ffn swiglu 内部公式 = silu(second_half) * first_half
         即 weight1 = cat([up_w.T, gate_w.T]) → silu(x@gate) * (x@up)
         inner_precise=1 (fp16 场景必填, 否则报错)
```

**统一入口**:
```python
def apply_fusion_ops(model):
    # 1. RMSNorm → npu_rms_norm
    # 2. RoPE rotate → npu_rotary_mul
    # 3. RoPE cos/sin → 图外预计算
    # 4. FFN → npu_ffn (swiglu)
```

### 2.4 profiling.py — Profiling 封装

**功能**: 先 warmup 触发图编译, 再正式采集 profiling 数据。

**流程**:
1. Warmup: 跑一次推理触发 `torch.compile` 编译, 不采集
2. `torch.npu.synchronize()`: 确保异步操作完成
3. `torch_npu.profiler.profile`: 采集 CPU + NPU 活动
4. `tensorboard_trace_handler`: 输出到目录

### 2.5 run_infer.py — 主入口

**功能**: CLI 入口, 支持 batch_size / seq_len / profiling / fusion 开关。

**用法**:
```bash
# 默认短文本推理
python -m qwen_varlen.run_infer

# 大输入 + profiling
python -m qwen_varlen.run_infer --batch-size 8 --seq-len 256 --profiling

# 对比基准 (禁用融合算子)
python -m qwen_varlen.run_infer --no-fusion --prof-dir ./prof_baseline

# 指定 NPU 设备
python -m qwen_varlen.run_infer --device 8
```

**执行流程**:
1. `load_model()`: 加载模型到 NPU, 注册 npu_fia, 应用融合算子
2. `prepare_varlen_inputs()`: token 拼接
3. `setup_varlen_attention()`: 注入 mask/cos/sin
4. `compile_model()`: torchair GE 图编译
5. Warmup + 正式推理 (或 profiling)

---

## 3. 已实现的优化点

### 3.1 Attention 融合 (基础)

| 优化 | 说明 |
|------|------|
| `npu_fused_infer_attention_score` | 替代手动 MatMul+Softmax+MatMul, 支持 varlen TND 布局 |
| GQA 原生支持 | 通过 `num_key_value_heads` 参数, 无需 repeat_kv |
| block-diagonal causal mask | `sparse_mode=3` + 固定 2048x2048 mask |
| dynamic=True | actual_seq_lengths 以 tensor 传入, 避免 dynamo specialize |

### 3.2 RMSNorm 融合

| 优化 | 说明 |
|------|------|
| `npu_rms_norm` | 8 个小算子 → 1 个融合算子 |
| 消除 Cast | 原始 RMSNorm 做 fp16→fp32→fp16 两次 Cast, 融合算子内部处理 |
| GE 自动融合 | 编译器自动将 `residual + RMSNorm` 融合为 `AddRmsNorm` |

### 3.3 RoPE 融合

| 优化 | 说明 |
|------|------|
| `npu_rotary_mul` | rotate_half(StridedSlice+Neg+Cat)+Mul+Add (12个) → 2 个融合算子 |
| 图外预计算 cos/sin | 消除图内 Cast(206us)+MatMul+Cat+Cos+Sin (5个 kernel) |
| fp16 直接注入 | cos/sin 在 Python 侧算完直接传 fp16 tensor 进图 |

### 3.4 FFN 融合

| 优化 | 说明 |
|------|------|
| `npu_ffn(swiglu)` | 2 MatMul + Cat + SwiGLu + MatMul (5个) → 1 个融合算子 |
| 预拼接权重 | gate_w + up_w 离线拼为 weight1, 避免运行时 Cat |
| inner_precise=1 | fp16 场景必须设置, 否则 aclnnFFNV2 报参数错误 |

### 3.5 Profiling 精确采集

| 优化 | 说明 |
|------|------|
| Warmup 隔离 | 先跑一次触发图编译, 不采集, 避免编译开销污染数据 |
| `torch.npu.synchronize()` | 确保 NPU 异步操作完成后退出采集区间 |

---

## 4. 性能现状

### 4.1 小输入对比 (batch=1, total=10 tokens)

| 指标 | 无融合 | v1融合 | v2融合 |
|------|--------|--------|--------|
| 总算子数 | 1,174 | 446 | 339 |
| NPU Computing (us) | 9,653 | 4,387 | 5,219 |
| Host ExecuteGraph (us) | 135,190 | 54,647 | 55,112 |

### 4.2 大输入对比 (batch=8, seq_len=276, total=2208 tokens)

| 指标 | 无融合 | 融合后 | 降幅 |
|------|--------|--------|------|
| 总算子数 | 1,174 | 339 | **-71%** |
| NPU Computing (us) | 28,180 | 20,640 | **-26.8%** |
| Host ExecuteGraph (us) | 167,496 | 101,142 | **-39.6%** |
| NPU 利用率 | 16.8% | 20.4% | +3.6pp |

### 4.3 融合前后算子分布对比 (大输入)

| 算子类型 | 无融合 (数量) | 融合后 (数量) | 变化 |
|---------|-------------|-------------|------|
| Mul | 220 | 0 | -220 |
| Cast | 151 | 0 | -151 |
| Add | 145 | 0 | -145 |
| StridedSliceV2 | 96 | 0 | -96 |
| Neg | 48 | 0 | -48 |
| Rsqrt | 49 | 0 | -49 |
| ReduceMean | 49 | 0 | -49 |
| Pow/Square | 48 | 0 | -48 |
| Swish | 24 | 0 | -24 |
| aclnnCat | 49 | 0 | -49 |
| Sin/Cos | 2 | 0 | -2 |
| aclnnMm | 96 | 24 | -72 |
| Transpose | 97 | 96 | -1 |
| aclnnAddmm | 72 | 72 | 0 |
| FusedInferAttentionScore | 24 | 24 | 0 |
| **新增融合算子** | | | |
| AddRmsNorm | 0 | 48 | +48 |
| RotaryMul | 0 | 48 | +48 |
| aclnnFFNV3 | 0 | 24 | +24 |
| **总计** | **1,174** | **339** | **-835** |

### 4.4 当前 NPU 计算耗时分布 (大输入, 融合后)

| 算子 | 数量 | 总耗时(us) | 占比 | 均值(us) |
|------|------|-----------|------|---------|
| aclnnFFNV3 | 24 | 9,367 | 45.4% | 390 |
| FusedInferAttentionScore | 24 | 3,556 | 17.2% | 148 |
| aclnnMmV3 (lm_head) | 1 | 2,276 | 11.0% | 2,276 |
| Transpose | 96 | 1,729 | 8.4% | 18 |
| aclnnAddmm | 72 | 1,157 | 5.6% | 16 |
| AddRmsNorm | 48 | 1,113 | 5.4% | 23 |
| aclnnMm | 24 | 724 | 3.5% | 30 |
| RotaryMul | 48 | 675 | 3.3% | 14 |
| 其余 | 2 | 44 | 0.2% | 22 |
| **总计** | **339** | **20,640** | 100% | |

### 4.5 Host-Device 流水线分析

```
ExecuteGraph (Host): 101,142 us
├── NPU 时间线:      64,083 us (63.4%)
│   ├── kernel 计算:  20,640 us (20.4%)  ← NPU 真正在算
│   └── kernel 等待:  43,450 us (42.9%)  ← NPU 空转等 Host 下发
└── 图启动/收尾:      37,059 us (36.6%)  ← GE session/输入组装/输出刷新
```

**主要瓶颈**: NPU 空闲等待占 42.9%, 其中 lm_head 的 wait=42,574us (等前面 24 层算完)。根因是 339 个 kernel 逐个同步下发, 每个 kernel 的 ACL API 调用 (GetWorkspaceSize + Malloc + Launch) 约 100-300us。

---

## 5. 尚未实施的优化点

### P1: lm_head 只算最后 token

| 指标 | 值 |
|------|-----|
| 当前耗时 | 2,276 us (单个 kernel, NPU Computing 的 11%) |
| 优化后 | ~1 us (只算 1 个位置 vs 2208 个) |
| 难度 | 低 |
| 方法 | 传 `logits_to_keep=1`, 只算最后位置 logits |
| 适用场景 | next-token prediction (当前场景) |

### P2: set_dim_gears 替代 dynamic=True

| 指标 | 值 |
|------|-----|
| 预期收益 | Host 开销降 50%+ |
| 难度 | 中 |
| 方法 | `torchair.inference.set_dim_gears` 预设 shape 分档, 编译为准静态图 |
| 原理 | 跳过动态 shape 的 workspace 计算, 减少 ACL 调用开销 |

### P3: enable_single_stream

| 指标 | 值 |
|------|-----|
| 预期收益 | 减少调度开销 |
| 难度 | 低 |
| 方法 | `config.ge_config.enable_single_stream = 1` |
| 原理 | 单 stream 避免多 stream 间同步 |

### P4: Transpose 融合

| 指标 | 值 |
|------|-----|
| 当前 | 96 个 Transpose (数量最多, 总耗时 1,729us) |
| 预期 | 减少到 ~48 个 |
| 难度 | 中 |
| 方法 | `npu_confusion_transpose` 融合 reshape+transpose |
| 来源 | q/k/v proj 的 view().transpose(1,2) (72个) + npu_fia 的 permute (24个) |

### P5: 多 stream 异步执行

| 指标 | 值 |
|------|-----|
| 预期收益 | 消除 Host-Device 流水线等待 |
| 难度 | 高 |
| 方法 | 多 stream 流水线, Host 下发 N+1 时 NPU 算第 N 次 |
| 原理 | 当前 kernel 串行下发, 每个等 ACL API 返回后才能下发下一个 |

---

## 6. 关键技术决策记录

### 6.1 为什么用 monkey-patch 而非继承

transformers 的 `AutoModelForCausalLM.from_pretrained` 直接实例化 `Qwen2ForCausalLM`, 继承需要修改 `auto_map` 或重写加载逻辑。Monkey-patch 直接替换类方法/模块函数, 零侵入, 对已加载模型立即生效。

### 6.2 为什么 actual_seq_lengths 用 tensor 传入

如果用 Python list, `torch.compile` 的 dynamo 会将 list 值 specialize 为编译时常量, 导致每个不同的 seq_len 组合都要重新编译。用 tensor 传入, dynamo 不会 specialize, 配合 `dynamic=True` 实现一次编译多 shape 复用。

### 6.3 为什么 cos/sin 图外预计算

原始 `Qwen2RotaryEmbedding.forward` 在图内计算 cos/sin, 其中 `cos.to(torch.float16)` 被编译为 NPU Cast kernel, profiling 显示耗时 206us。图外在 Python 侧算完直接传 fp16 tensor 进图, 消除了 5 个图内 kernel (Cast + MatMul + Cat + Cos + Sin)。

### 6.4 为什么 npu_ffn 的 weight1 = cat([up_w.T, gate_w.T])

通过实验验证 (非文档): `npu_ffn` 的 swiglu 内部公式实际是 `silu(second_half) * first_half`, 与文档描述的 `swish(A) * B` 相反。因此:
- weight1 前半 = up_w.T → first_half = x@up_w (被乘)
- weight1 后半 = gate_w.T → second_half = x@gate_w (被 silu)
- 结果 = silu(x@gate_w) * (x@up_w) = 原始 SwiGLU

### 6.5 为什么需要 warmup

`torch.compile` + `torchair` GE 图模式首次执行触发编译: Python trace → FX graph → GE graph → 编译 → 加载到 NPU。编译耗时数十秒, 必须先 warmup 排除, 否则 profiling 数据被编译开销严重污染。

---

## 7. 文件清单与行数

| 文件 | 行数 | 职责 |
|------|------|------|
| `__init__.py` | 15 | 包入口, 导出公共接口 |
| `attention.py` | 62 | npu_fused_infer_attention_score 注册 + causal mask 构建 |
| `varlen_utils.py` | 82 | varlen 输入拼接 + cos/sin 预计算 + attention 参数注入 |
| `fusion_ops.py` | 166 | 4 项融合算子替换 (RMSNorm/RoPE rotate/RoPE cos-sin/FFN) |
| `profiling.py` | 39 | warmup + profiling 采集封装 |
| `run_infer.py` | 134 | CLI 入口, 模型加载→编译→推理→profiling |
| **总计** | **498** | |
