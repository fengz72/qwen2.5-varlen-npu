"""
变长 (varlen) 输入处理 — token 拼接与位置编码

通用工具, 不依赖具体模型结构。
"""

import torch


def generate_varlen_inputs(batch_size, seq_len):
    """直接生成全 0 token ids 的 varlen 输入, 不需要 tokenizer。

    Args:
        batch_size:  序列条数
        seq_len:     每条序列的 token 数

    Returns:
        concat_ids:  [1, total_len] 全 0 token ids
        concat_pos:  [1, total_len] 拼接后的 position ids
        seq_lens:    list[int] 每条序列的长度
        cum_seq_lens: list[int] 累积长度 (用于 actual_seq_lengths)
    """
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


def generate_compact_varlen_inputs(batch_size, seq_len, prefix_len):
    """生成 prefix sharing 的 compact varlen 输入 (全 0 token, 不需要 tokenizer)。

    compact layout: [prefix, req_0, req_1, ..., req_n]
    每条 seq 包含 prefix, compact 中 prefix 只存一份。
    T_compact = prefix_len + batch_size * (seq_len - prefix_len)

    position_ids 仍是 expanded 格式 [0..S-1, 0..S-1, ...] (用于 cos/sin gather)。

    Returns:
        compact_ids:  [1, T_compact] 全 0 token ids
        expanded_pos: [1, T_expanded] 拼接后的 position ids (expanded)
        seq_lens:     list[int] 每条序列的长度 (含 prefix)
        cum_seq_lens: list[int] 累积长度 (expanded, 用于 actual_seq_lengths)
    """
    seq_lens = [seq_len] * batch_size
    req_tokens = seq_len - prefix_len
    total_compact = prefix_len + batch_size * req_tokens
    compact_ids = torch.zeros(total_compact, dtype=torch.long).unsqueeze(0)

    pos_ids = [torch.arange(seq_len) for _ in range(batch_size)]
    expanded_pos = torch.cat(pos_ids).unsqueeze(0)

    cum_seq_lens = []
    acc = 0
    for s in seq_lens:
        acc += s
        cum_seq_lens.append(acc)
    return compact_ids, expanded_pos, seq_lens, cum_seq_lens
