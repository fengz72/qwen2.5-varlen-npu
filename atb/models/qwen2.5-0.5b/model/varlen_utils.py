"""
变长 (varlen) 模型相关处理 — attention 参数注入、cos/sin 预计算, prefix sharing index 构建。

通用 token 拼接工具已提取到 atb.tools.varlen。
"""

import torch

from .attention import build_causal_mask_2048


# ==================== Prefix sharing index 构建 ====================

def build_expand_index(prefix_len, seq_lens, device):
    """构建 expand index: compact → expanded 的位置映射。

    compact:  [prefix, req_0, req_1, ..., req_n]
    expanded: [prefix, req_0, prefix, req_1, ..., prefix, req_n]

    index[i] = expanded 位置 i 对应的 compact 位置
    返回长度 = T_expanded = sum(seq_lens)
    """
    indices = []
    compact_req_offset = prefix_len
    for sl in seq_lens:
        req_tokens = sl - prefix_len
        indices.append(torch.arange(prefix_len, device=device))
        indices.append(torch.arange(compact_req_offset, compact_req_offset + req_tokens, device=device))
        compact_req_offset += req_tokens
    return torch.cat(indices).to(torch.int64)


def build_restore_index(prefix_len, seq_lens, device):
    """构建 restore index: expanded → compact 的位置映射。

    compact:  [prefix, req_0, req_1, ..., req_n]
    expanded: [prefix, req_0, prefix, req_1, ..., prefix, req_n]

    index[i] = compact 位置 i 对应的 expanded 位置
    返回长度 = T_compact = prefix_len + sum(seq_lens - prefix_len)
    """
    prefix_idx = torch.arange(prefix_len, device=device)
    req_indices = []
    block_start = 0
    for sl in seq_lens:
        req_tokens = sl - prefix_len
        req_start = block_start + prefix_len
        req_indices.append(torch.arange(req_start, req_start + req_tokens, device=device))
        block_start += sl
    return torch.cat([prefix_idx] + req_indices).to(torch.int64)


def build_compact_last_indices(prefix_len, seq_lens, device):
    """构建 compact last indices: 每个请求最后一个 token 在 compact 中的位置。

    compact: [prefix, req_0, req_1, ..., req_n]
    返回长度 = n = len(seq_lens)
    """
    indices = []
    compact_offset = prefix_len
    for sl in seq_lens:
        req_tokens = sl - prefix_len
        indices.append(compact_offset + req_tokens - 1)
        compact_offset += req_tokens
    return torch.tensor(indices, dtype=torch.int64, device=device)


def precompute_rope_cos_sin(model, total_len, device):
    """图外预计算 RoPE 的 cos/sin, 直接生成 fp16, 避免图内 Cast kernel。

    Qwen2RotaryEmbedding.forward 原始计算:
        inv_freq @ position_ids (fp32) → cat → cos → sin → Cast(fp32→fp16)
    其中 Cast 在 profiling 中耗时 206us。

    图外预计算后, cos/sin 作为 fp16 tensor 注入, 图内不再有 Cast/MatMul/Cos/Sin。
    直接内联计算, 不依赖 rotary_emb.forward (可能已被 monkey-patch)。
    """
    rotary_emb = model.model.rotary_emb
    position_ids = torch.arange(total_len, dtype=torch.long, device=device).unsqueeze(0)

    inv_freq = rotary_emb.inv_freq
    scaling = rotary_emb.attention_scaling

    inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(device)
    position_ids_expanded = position_ids[:, None, :].float()
    freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = (emb.cos() * scaling).to(dtype=torch.float16)
    sin = (emb.sin() * scaling).to(dtype=torch.float16)

    rotary_emb._cached_cos = cos
    rotary_emb._cached_sin = sin


def setup_varlen_attention(model, cum_seq_lens, device):
    """向模型每一层注入 varlen attention 所需的 actual_seq_lengths 和 mask。

    同时禁用 transformers 自带的 _update_causal_mask (由推理算子内部处理)。
    预计算 RoPE cos/sin 并注入, 避免图内 Cast。

    Returns:
        atten_mask: 构建的因果掩码张量
    """
    atten_mask = build_causal_mask_2048(device)
    asl_tensor = torch.tensor(cum_seq_lens, dtype=torch.int64, device=device)
    for layer in model.model.layers:
        layer.self_attn.actual_seq_lengths_tensor = asl_tensor
        layer.self_attn.register_buffer('atten_mask', atten_mask)
    model.model._update_causal_mask = lambda *a, **kw: None

    # 图外预计算 cos/sin
    total_len = cum_seq_lens[-1] if cum_seq_lens else 0
    precompute_rope_cos_sin(model, total_len, device)

    return atten_mask
