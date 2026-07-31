"""
lm_head vocab 剪裁工具 — 通用 CausalLM 操作, 不依赖具体模型结构
"""

import json

import torch
import torch.nn as nn


def load_target_tokens(json_path):
    """加载 target token JSON, 返回 token_ids 列表。"""
    with open(json_path) as f:
        data = json.load(f)
    token_ids = data["token_ids"]
    assert len(token_ids) > 0, "token_ids 不能为空"
    print(f"[prune] target token_ids: {token_ids}")
    return token_ids


def prune_lm_head(model, token_ids):
    """裁剪 lm_head 仅保留 token_ids 对应的行。

    输出维度从 vocab_size 降为 len(token_ids)。
    若模型 tie_word_embeddings=True, 先 clone 解绑再裁剪, 不影响 embed_tokens。
    """
    if getattr(model.config, 'tie_word_embeddings', False):
        model.lm_head.weight = nn.Parameter(
            model.model.embed_tokens.weight.data.clone()
        )
        print("[prune] 解除 lm_head/embed_tokens weight tying")

    device = model.lm_head.weight.device
    target_ids_t = torch.tensor(token_ids, dtype=torch.long, device=device)
    pruned_weight = model.lm_head.weight.data.index_select(0, target_ids_t).clone()

    hidden_size = model.config.hidden_size
    new_lm_head = nn.Linear(hidden_size, len(token_ids), bias=False)
    new_lm_head.weight = nn.Parameter(pruned_weight)
    model.lm_head = new_lm_head.npu().half()
    model.config.vocab_size = len(token_ids)

    print(f"[prune] lm_head: [{len(token_ids)}, {hidden_size}]")
