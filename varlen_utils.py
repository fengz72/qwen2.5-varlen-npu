"""
变长 (varlen) 输入处理工具 — token 拼接、位置编码、attention 参数注入、cos/sin 预计算
"""

import torch

from .attention import build_causal_mask_2048


def prepare_varlen_inputs(tokenizer, input_texts):
    """将多条文本拼接为 varlen 格式的输入。

    Returns:
        concat_ids:  [1, total_len] 拼接后的 token ids
        concat_pos:  [1, total_len] 拼接后的 position ids
        seq_lens:    list[int] 每条文本的长度
        cum_seq_lens: list[int] 累积长度 (用于 actual_seq_lengths)
    """
    all_ids, pos_ids = [], []
    for text in input_texts:
        ids = tokenizer(text, return_tensors="pt")["input_ids"][0]
        all_ids.append(ids)
        pos_ids.append(torch.arange(ids.shape[0]))
    seq_lens = [x.shape[0] for x in all_ids]
    concat_ids = torch.cat(all_ids).unsqueeze(0)
    concat_pos = torch.cat(pos_ids).unsqueeze(0)
    cum_seq_lens = []
    acc = 0
    for s in seq_lens:
        acc += s
        cum_seq_lens.append(acc)
    return concat_ids, concat_pos, seq_lens, cum_seq_lens


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
        layer.self_attn.atten_mask = atten_mask
    model.model._update_causal_mask = lambda *a, **kw: None

    # 图外预计算 cos/sin
    total_len = cum_seq_lens[-1] if cum_seq_lens else 0
    precompute_rope_cos_sin(model, total_len, device)

    return atten_mask
