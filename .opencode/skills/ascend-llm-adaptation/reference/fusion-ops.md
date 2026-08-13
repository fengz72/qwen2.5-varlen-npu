# 算子融合实施

> 本文件按 SKILL.md Step 2.5 的算子融合分支组织。
> 每类算子的实现按分支列出，Qwen2.5 为已验证示例，其他分支待验证后补充。

## 算子融合分支总览

| 算子类型 | 分支选项 | 已验证 |
|---------|---------|--------|
| Norm | RMSNorm→`npu_rms_norm` / LayerNorm→`npu_layer_norm` | ✓ RMSNorm |
| 位置编码 | RoPE→图外预计算+`npu_apply_rotary_pos_emb` / 绝对位置→保留图内 / 无→跳过 | ✓ RoPE |
| Attention | `npu_fused_infer_attention_score`（推理算子） | ✓ |
| FFN | SwiGLU→可试 `npu_ffn`(需profile验证) / 拆分MatMul / 其他 | ✓ 拆分 |
| QKV proj | 融合 / 不融合 | ✓ 不融合 |

> 后续验证新分支后，在此补全代码。

## monkey-patch 模板

> 占位符说明：`<modeling_module>` 是目标模型的 transformers modeling 模块
>（如 `modeling_qwen2`、`modeling_llama`），`<ModelClass>` / `<function_name>`
> 替换为目标模型对应的类/函数名。

```python
import torch_npu
from transformers.models.<model_module> import <modeling_module>

def apply_fusion_ops():
    # 1. RMSNorm → npu_rms_norm
    <modeling_module>.<ModelRMSNorm>.forward = lambda self, x: \
        torch_npu.npu_rms_norm(x, self.weight, self.variance_epsilon)[0]

    # 2. RoPE rotate → npu_apply_rotary_pos_emb
    <modeling_module>.<apply_rotary_pos_emb> = _npu_apply_rotary_pos_emb

    # 3. RoPE cos/sin → return precomputed (eliminates 5 graph kernels)
    <modeling_module>.<ModelRotaryEmbedding>.forward = lambda self, x, pos: \
        (self._cached_cos, self._cached_sin)
```

## npu_apply_rotary_pos_emb 实现

```python
def _npu_apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """用 npu_apply_rotary_pos_emb 融合算子替代 2×npu_rotary_mul。

    q/k 为 TND [T, N, D] 格式。cos/sin 为 [1, T, D]，转换为 [T, 1, D]。
    一次调用同时处理 Q 和 K，原生支持 3D TND 布局。

    注意: GE 图编译时 ApplyRotaryPosEmb 可能返回 4D [1,T,N,D] 而非 3D [T,N,D]，
    需显式 reshape 回输入 shape，否则下游 FIA 的 num_heads 推断错误。
    """
    cos = cos.squeeze(0).unsqueeze(1)   # [1, T, D] → [T, 1, D]
    sin = sin.squeeze(0).unsqueeze(1)
    q_embed, k_embed = torch_npu.npu_apply_rotary_pos_emb(
        q, k, cos, sin, layout="TND", rotary_mode="half"
    )
    q_embed = q_embed.reshape(-1, q.size(1), q.size(2))
    k_embed = k_embed.reshape(-1, k.size(1), k.size(2))
    return q_embed, k_embed
```

## cos/sin 图外预计算（RoPE 分支，✓ Qwen2.5 已验证）

一旦 patch `RotaryEmbedding.forward` 返回缓存值，就必须在图外预计算
cos/sin 并注入，否则图中没有位置编码计算。

> 占位符说明：`<base_model>` 是内层模型（如 `model.model`），
> `<rotary_emb>` 是 rotary embedding 属性名，`<scaling_attr>` 是 scaling 属性名。

```python
import torch

def precompute_rope_cos_sin(model, total_len, device):
    rotary_emb = model.<base_model>.<rotary_emb>
    position_ids = torch.arange(total_len, dtype=torch.long, device=device).unsqueeze(0)
    inv_freq = rotary_emb.inv_freq
    scaling = rotary_emb.<scaling_attr>  # e.g., attention_scaling in Qwen2

    inv_freq_expanded = inv_freq[None, :, None].float().expand(1, -1, 1).to(device)
    position_ids_expanded = position_ids[:, None, :].float()
    freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = (emb.cos() * scaling).to(dtype=torch.float16)
    sin = (emb.sin() * scaling).to(dtype=torch.float16)

    rotary_emb._cached_cos = cos
    rotary_emb._cached_sin = sin
```

消除 5 个图内 kernel：Cast(fp32→fp16, 206us) + MatMul + Cat + Cos + Sin。

## Qwen2.5-0.5b 融合实践（✓ 已验证示例）

| 算子 | 决策 | 原因 |
|------|------|------|
| RMSNorm | **融合** → `npu_rms_norm` | 8→1 ops，无 tiling 问题 |
| RoPE rotate | **融合** → `npu_apply_rotary_pos_emb` | 原生 TND 支持 |
| RoPE cos/sin | **图外预计算** | 消除 5 个图内 kernel（含 206us Cast） |
| Attention | **融合** → `npu_fused_infer_attention_score` | 推理算子，原生 GQA + varlen |
| FFN (MLP) | **不融合** | profiling 发现 `npu_ffn` MAC 仅 43.5%，拆分后 85%，快 16.4% |
| QKV projection | **不融合** | StridedSliceV3 开销 > MatMul 节省 |

### FFN 融合优化分析

> 测试环境: Ascend910_9382, batch=10, seq_len=208, 单线程

`npu_ffn` (FFNV3) 占总执行时间 54%，MAC 利用率仅 43.5%。

根因：`npu_ffn` 内部处理两个不同 shape 的 MatMul:
- MatMul1: [2080,896] @ [896,9728] → [2080,9728] (K=896, N=9728)
- MatMul2: [2080,4864] @ [4864,896] → [2080,896] (K=4864, N=896)

两个 MatMul 的 K 和 N 差异大，融合 kernel 需要找一种 tiling 同时适配两者，
导致妥协。拆分后每个 MatMul 独立 tiling:
- gate/up: K=896, N=4864 → MAC 85.4%
- down: K=4864, N=896 → MAC 83.4%

拆分后 MLP 快 16.4%，全模型快 11.0%。中间量 HBM 往返的代价远小于
tiling 优化的收益。Swish+Mul 被 GE 自动融合为 EltwiseBroadcastFusionOp，
额外开销仅 884us/iter，可忽略。
