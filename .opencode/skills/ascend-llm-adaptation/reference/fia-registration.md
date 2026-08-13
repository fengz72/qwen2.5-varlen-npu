# NPU FIA 注册

> 本文件按 SKILL.md Step 1.2 的 attention 模式分支组织代码。
> 每个分支独立，按 Step 1.0 画像选择对应实现。

## Attention 模式分支总览

| 模式 | sparse_mode | mask | 已验证 |
|------|-------------|------|--------|
| causal | 3 (rightDownCausal) | 上三角 bool | ✓ Qwen2.5 |
| bidirectional | 待补充 | 无需 mask | 待验证 |
| pair | 待补充 | block-diagonal | 待验证 |

> 后续验证新分支后，在此补全代码。

## 通用注册框架（所有分支共用）

```python
import torch
import torch_npu
from torchair.ops import npu_fused_infer_attention_score as _torchair_fia
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

# 关键：编译时用 torchair.ops，eager 时用 torch_npu。
# torchair.ops 为 dynamo_export 生成正确图节点，
# torch_npu 支持 eager 执行用于 golden 生成。
# 此模式对所有 torchair.ops 算子通用。
def _call_fia(q_t, k_t, v_t, **fia_kwargs):
    if torch.compiler.is_compiling():
        out, _ = _torchair_fia(q_t, k_t, v_t, **fia_kwargs)
    else:
        out, _ = torch_npu.npu_fused_infer_attention_score(q_t, k_t, v_t, **fia_kwargs)
    return out

# BNSD → TND 转换（varlen 场景需要，供推理算子使用）
def _to_tnd(q):
    return q.permute(0, 2, 1, 3).squeeze(0).contiguous()
```

## causal 分支（✓ Qwen2.5 已验证）

```python
def npu_fia_causal_forward(module, query, key, value, attention_mask,
                           dropout=0.0, scaling=1.0, sliding_window=None, **kwargs):
    n = int(query.shape[1])
    n_kv = int(key.shape[1])
    q_t = _to_tnd(query)
    k_t = _to_tnd(key)
    v_t = _to_tnd(value)

    out = _call_fia(q_t, k_t, v_t,
        num_heads=n,
        input_layout="TND",
        scale=scaling,
        actual_seq_lengths=module.actual_seq_lengths_tensor,
        actual_seq_lengths_kv=module.actual_seq_lengths_tensor,
        num_key_value_heads=n_kv,          # GQA 原生支持，无需手动 repeat_kv
        atten_mask=module.atten_mask,
        sparse_mode=3,                      # rightDownCausal
    )
    out = out.unsqueeze(0).contiguous()
    return out, None

ALL_ATTENTION_FUNCTIONS.register("npu_fia", npu_fia_causal_forward)
```

### 因果掩码构建

```python
import torch

def build_causal_mask_2048(device):
    """构建 2048×2048 上三角因果掩码 (bool)，用于 sparse_mode=3。"""
    return torch.triu(
        torch.ones(2048, 2048, dtype=torch.bool, device=device), diagonal=1
    )
```

### 要点说明

- `sparse_mode=3`（rightDownCausal）+ 固定 2048×2048 bool mask，让算子根据
  `actual_seq_lengths` 内部生成 block-diagonal 因果掩码。
- GQA 通过 `num_key_value_heads` 原生支持，无需手动 `repeat_kv`。
- `actual_seq_lengths_tensor` 和 `atten_mask` 需在导出前注入到每一层。
  eager 验证时也需手动设置。

## bidirectional 分支（待验证）

> 适用场景：编码器(BERT)、embedding 提取等无需因果掩码的场景。
> sparse_mode=0 或不传 atten_mask，varlen 场景仍需 actual_seq_lengths
> 防止序列间信息泄露。待验证后补全代码。

## pair 分支（待验证）

> 适用场景：句对匹配、搜索相关性打分等 [query, doc] 拼接输入。
> 需自定义 block-diagonal mask，控制 query 和 doc 间的 attention 方向。
> 待验证后补全代码。
