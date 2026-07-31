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
