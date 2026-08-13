# 常见问题

> 按适配阶段分组，便于按出错位置快速定位。

## Eager / Export 阶段

### 图中出现 Pack/Unpack 算子
**现象：** GE 图中 Linear 层附近有 Pack/Unpack/Reshape 节点。
**原因：** hidden_states 有 batch 维度，或 reshape 使用了 SymInt。
**修复：** 保持 hidden_states 为 2D `[T, D]`，使用 `reshape(-1, known_dim)`。

### eager 模式下 npu_fia 找不到
**现象：** `npu_fused_infer_attention_score` 在 golden 生成时失败。
**原因：** eager 模式下使用了 `torchair.ops` 版本（仅编译时可用）。
**修复：** 用 `torch.compiler.is_compiling()` 分支区分。

### OM 未编码动态 shape
**现象：** OM 只接受固定 shape 输入。
**原因：** `dynamo_export` 前缺少 `mark_dynamic` 调用。
**修复：** 用 `torch._dynamo.mark_dynamic` 显式标记 T 和 N 维度。

### RoPE cos/sin Cast kernel（206us）
**现象：** profiling 显示 rotary embedding 中有昂贵的 Cast(fp32→fp16)。
**原因：** RotaryEmbedding 在 fp32 下计算 cos/sin 再 cast。
**修复：** 图外预计算 fp16 cos/sin，作为缓存 tensor 注入。

## ATC 编译阶段

### NZ pass 未生效
**现象：** ATC 输出仍然是 MatMul（非 MatMulV3）。
**原因：** pass `.so` 不在 vendor 目录，或权重非 Const 节点。
**修复：** 确认导出配置 `frozen_parameter=1`；检查 vendor 目录路径。

### GE 内部格式属性缺失
**现象：** MatMulV3 节点输出错误或崩溃。
**原因：** 手动创建节点时缺少格式属性。
**修复：** 设置 `input_desc_attr_format_for_int`、`_cube_vector_core_type` 等。

### NZ pass 没有开关
**现象：** 安装 pass 后无法获得"无 NZ"baseline。
**原因：** pass 安装后对所有 ATC 编译自动生效。
**修复：** 在 Step 4（安装前）获取 baseline。如需移除，从 vendor 目录删除 `.so`。

## 运行时 / Profiling 阶段

### FFN 融合比预期慢
**现象：** `npu_ffn` 比拆分 MatMul 慢，尽管算子更少。
**原因：** FFNV3 对两个不同 MatMul shape 做 tiling 妥协。
**修复：** 不融合 FFN。用 profiling 确认 — 检查 `aic_mac_ratio`。
