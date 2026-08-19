"""
变长 (varlen) 模型相关处理 — attention 参数注入、cos/sin 预计算

通用 token 拼接工具已提取到 atb.tools.varlen。
"""

import torch

from .attention import build_causal_mask_2048


def precompute_rope_cos_sin(model, max_seq_len, device):
    """预计算 RoPE cos/sin 表并注册为 buffer, 配合 frozen_parameter 成为图常量。

    cos/sin 表 [1, max_seq_len, 64] 作为 register_buffer 注册,
    图内 _npu_rotary_emb_forward 按 position_ids gather 出 [1, T, 64]。
    """
    rotary_emb = model.model.rotary_emb
    position_ids = torch.arange(max_seq_len, dtype=torch.long, device=device).unsqueeze(0)

    inv_freq = rotary_emb.inv_freq
    scaling = rotary_emb.attention_scaling

    inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(device)
    position_ids_expanded = position_ids[:, None, :].float()
    freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = (emb.cos() * scaling).to(dtype=torch.float16)
    sin = (emb.sin() * scaling).to(dtype=torch.float16)

    rotary_emb.register_buffer('_cos_table', cos, persistent=False)
    rotary_emb.register_buffer('_sin_table', sin, persistent=False)


def setup_varlen_attention(model, cum_seq_lens, device, max_seq_len=1024):
    """向模型每一层注入 varlen attention 所需的 actual_seq_lengths 和 mask。

    同时禁用 transformers 自带的 _update_causal_mask (由推理算子内部处理)。
    预计算 RoPE cos/sin 表并注册为 buffer (图常量)。

    Returns:
        atten_mask: 构建的因果掩码张量
    """
    atten_mask = build_causal_mask_2048(device)
    asl_tensor = torch.tensor(cum_seq_lens, dtype=torch.int64, device=device)
    for layer in model.model.layers:
        layer.self_attn.actual_seq_lengths_tensor = asl_tensor
        layer.self_attn.register_buffer('atten_mask', atten_mask)
    model.model._update_causal_mask = lambda *a, **kw: None

    precompute_rope_cos_sin(model, max_seq_len, device)

    return atten_mask
