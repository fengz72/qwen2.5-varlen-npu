# 图信息决策

> 本文件按 SKILL.md Step 3 的分支组织。
> 序列拼接模式、输出设计、lm_head 处理各有分支表，Qwen2.5 为已验证示例。

## 序列拼接模式总览

| 模式 | hidden_states | TND patch | ASL 注入 | 已验证 |
|------|--------------|-----------|---------|--------|
| varlen | 2D [T, D] | 需要 | 需要 | ✓ Qwen2.5 |
| 固定 batch | 3D [B,S,D] 或 2D | 可选 | 不需要 | 待验证 |
| 单序列 | 2D [S, D] | 不需要 | 不需要 | 待验证 |

> 后续验证新分支后，在此补全代码。

## 输出设计总览

| 输出模式 | forward 返回 | 已验证 |
|---------|-------------|--------|
| 末 token logits | `lm_head(last_hidden)` | ✓ Qwen2.5 |
| 全 token logits | `lm_head(all_hidden)` | 待验证 |
| embedding | `norm(hidden)` + pooling | 待验证 |
| 自定义 head | `custom_head(hidden)` | 待验证 |
| KV cache | `hidden + kv_cache` | 待验证 |

> 后续验证新分支后，在此补全代码。

## 2D TND patch（varlen 分支，✓ Qwen2.5 已验证）

> 占位符说明：`<ModelAttention>` 是目标模型的 Attention 类
>（如 `Qwen2Attention`、`LlamaAttention`）。需适配 forward 签名以匹配
> 模型的 attention 接口。

```python
<modeling_module>.<ModelAttention>.forward = _patched_attention_forward

def _patched_attention_forward(self, hidden_states, position_embeddings,
                               attention_mask, past_key_values=None, **kwargs):
    num_heads = int(self.config.num_attention_heads)
    head_dim = int(self.head_dim)
    # hidden_states 为 [T, D] — Linear 直接在 2D 上工作
    q = self.q_proj(hidden_states).reshape(-1, num_heads, head_dim)
    k = self.k_proj(hidden_states).reshape(-1, num_kv_heads, head_dim)
    v = self.v_proj(hidden_states).reshape(-1, num_kv_heads, head_dim)
    # ... apply_rotary_pos_emb + attention 调用 ...
    attn_output = attn_output.reshape(-1, hidden_size).contiguous()
    return self.o_proj(attn_output), None
```

**规则：** 所有 `reshape` 必须使用 Python `int` 常量 + `-1`，不能提取
`SymInt`。这能防止 GE 生成动态 shape Pack 节点。

## varlen token 拼接（varlen 分支，✓ Qwen2.5 已验证）

```python
import torch

def generate_varlen_inputs(batch_size, seq_len):
    seq_lens = [seq_len] * batch_size
    concat_ids = torch.zeros(batch_size * seq_len, dtype=torch.long).unsqueeze(0)
    pos_ids = [torch.arange(seq_len) for _ in range(batch_size)]
    concat_pos = torch.cat(pos_ids).unsqueeze(0)
    cum_seq_lens = []
    acc = 0
    for s in seq_lens:
        acc += s
        cum_seq_lens.append(acc)
    return concat_ids, concat_pos, seq_lens, cum_seq_lens
```

## varlen attention 参数注入（varlen 分支，✓ Qwen2.5 已验证）

> 占位符说明：`<base_model>` 是内层模型（如 `model.model`），
> `<layers>` 是层列表，`<_update_causal_mask>` 是 causal mask 方法名。

```python
import torch

def setup_varlen_attention(model, cum_seq_lens, device):
    atten_mask = torch.triu(
        torch.ones(2048, 2048, dtype=torch.bool, device=device), diagonal=1
    )
    asl_tensor = torch.tensor(cum_seq_lens, dtype=torch.int64, device=device)
    for layer in model.<base_model>.<layers>:
        layer.self_attn.actual_seq_lengths_tensor = asl_tensor
        layer.self_attn.register_buffer('atten_mask', atten_mask)
    model.<base_model>.<_update_causal_mask> = lambda *a, **kw: None
    precompute_rope_cos_sin(model, cum_seq_lens[-1], device)
    return atten_mask
```

## 动态 shape 标记

```python
torch._dynamo.mark_dynamic(input_ids, 0)            # [T] → T 动态
torch._dynamo.mark_dynamic(actual_seq_lengths, 0)   # [N] → N 动态
torch._dynamo.mark_dynamic(cos, 1)                  # [1, T, 64] → T 动态
torch._dynamo.mark_dynamic(sin, 1)                  # [1, T, 64] → T 动态
torch._dynamo.mark_static(cos, 0)                   # B=1 固定
torch._dynamo.mark_static(cos, 2)                   # D=64 固定
```

## ExportWrapper（末 token logits 分支，✓ Qwen2.5 已验证）

> 占位符说明：`<base_model>` 是内层模型，`<layers>`/`<rotary_emb>`/
> `<embed_tokens>`/`<norm>`/`<lm_head>` 是属性名 — 替换为目标模型的结构。

```python
import torch
import torch.nn as nn

class ExportWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, position_ids, actual_seq_lengths, cos, sin):
        # 注入 varlen 参数 + 缓存 cos/sin
        for layer in self.model.<base_model>.<layers>:
            layer.self_attn.actual_seq_lengths_tensor = actual_seq_lengths
        self.model.<base_model>.<rotary_emb>._cached_cos = cos
        self.model.<base_model>.<rotary_emb>._cached_sin = sin

        m = self.model.<base_model>
        hidden = m.<embed_tokens>(input_ids)
        position_embeddings = m.<rotary_emb>(hidden, position_ids)
        for layer in m.<layers>:
            hidden = layer(hidden, attention_mask=None,
                           position_embeddings=position_embeddings,
                           position_ids=position_ids, use_cache=False)
        # 仅提取每条序列最后一个 token
        last_indices = actual_seq_lengths - 1
        last_hidden = hidden.index_select(0, last_indices)
        last_hidden = m.<norm>(last_hidden)
        return self.model.<lm_head>(last_hidden)
```

启用 `frozen_parameter=1` 后，图输入缩减为：

| # | 名称 | Shape | Dtype | 动态维度 |
|---|------|-------|-------|---------|
| 0 | actual_seq_lengths | `[N]` | int64 | dim 0 |
| 1 | cos | `[1, T, 64]` | float16 | dim 1 |
| 2 | sin | `[1, T, 64]` | float16 | dim 1 |
| 3 | input_ids | `[T]` | int64 | dim 0 |

`position_ids` 虽是 forward 参数，但因 cos/sin 图外预计算而在图中消除。

## lm_head 词表裁剪（裁剪分支，✓ Qwen2.5 已验证）

```python
# 克隆 lm_head 权重，index_select 目标行，替换为新的 Linear
# 如 tie_word_embeddings=True，先 clone 解绑再裁剪
prune_lm_head(model, target_token_ids)  # lm_head 替换为 [hidden, len(ids)]
```
