"""
为 AIR 导出的 OM 模型生成 7 个用户输入数据 + golden logits。

prefix sharing 模式 (prefix_len > 0):
  compact input_ids [T_c] = [prefix, req_1, ..., req_n] (省存储)
  attention 内部 expand→QKV→Attn→restore (图内处理)

frozen_parameter=1 后, FFN 权重和 atten_mask 已冻结为图常量,
OM 有 7 个图输入 (Data 节点):
  - actual_seq_lengths   [N] int64    — expanded 累积序列长度
  - cos                  [1, T_e, 64] fp16 — RoPE cos (expanded)
  - sin                  [1, T_e, 64] fp16 — RoPE sin (expanded)
  - input_ids            [T_c] int64  — compact token ids
  - expand_index         [T_e] int64  — compact→expanded 映射
  - restore_index        [T_c] int64  — expanded→compact 映射
  - compact_last_indices [N] int64    — 每条序列最后 token 在 compact 中的位置

cos/sin 预计算策略:
  1. 图外预计算 max_seq_len 的 cos/sin 表 [1, MAX_SEQ_LEN, 64] (一次)
  2. 运行时按 expanded position_ids gather: cos_table[:, pos, :] → [1, T_e, 64]
     varlen 中每条 seq 的 position 从 0 重启: pos = [0..S-1, 0..S-1, ...]

同时运行 eager 模式生成 golden logits 供精度对比。
"""

import argparse
import json
import os
import torch
import torch_npu

from .varlen_utils import setup_varlen_attention, precompute_rope_cos_sin
from .varlen_utils import build_expand_index, build_restore_index, build_compact_last_indices
from .export_air import load_model, ExportWrapper
from atb.tools.varlen import generate_compact_varlen_inputs, generate_varlen_inputs
from atb.tools.lm_head_prune import load_target_tokens, prune_lm_head

_MODEL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL_PATH = "/export/home/models/Qwen2.5-0.5B"
DEFAULT_OUTPUT_DIR = os.path.join(_MODEL_DIR, "input_data")
DEFAULT_TARGET_TOKEN_FILE = os.path.join(_MODEL_DIR, "target_tokens.json")
DEFAULT_DEVICE = 0

BATCH_SIZE = 10
SEQ_LEN = 208
PREFIX_LEN = 25
MAX_SEQ_LEN = 2048


def main():
    parser = argparse.ArgumentParser(description="生成 OM 输入数据 + golden logits")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="模型路径")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="输出目录")
    parser.add_argument("--device", type=int, default=DEFAULT_DEVICE, help="NPU 设备号")
    parser.add_argument("--prefix-len", type=int, default=PREFIX_LEN, help="共享前缀长度 (0 则不启用)")
    parser.add_argument("--prune-lm-head", action="store_true",
                        help="开启 lm_head vocab 剪裁 (与 export_air.py --prune-lm-head 一致)")
    parser.add_argument("--target-token-file", default=DEFAULT_TARGET_TOKEN_FILE,
                        help=f'target token JSON 文件 (默认: {DEFAULT_TARGET_TOKEN_FILE})')
    args = parser.parse_args()

    prefix_len = args.prefix_len

    # 1. 加载模型 (与 export 完全一致)
    model = load_model(args.model_path, args.device)

    # 1.1 lm_head vocab 剪裁 (与 export_air.py 一致)
    token_ids = None
    if args.prune_lm_head:
        if not args.target_token_file or not os.path.exists(args.target_token_file):
            raise FileNotFoundError(f"target_token_file 不存在: {args.target_token_file}")
        token_ids = load_target_tokens(args.target_token_file)
        prune_lm_head(model, token_ids)

    # 2. 准备 varlen 输入
    if prefix_len > 0:
        compact_ids, expanded_pos, seq_lens, cum_seq_lens = generate_compact_varlen_inputs(
            BATCH_SIZE, SEQ_LEN, prefix_len
        )
        t_expanded = cum_seq_lens[-1]
        t_compact = compact_ids.shape[1]
        print(f"prefix sharing: prefix_len={prefix_len}, T_compact={t_compact}, T_expanded={t_expanded}")
    else:
        compact_ids, expanded_pos, seq_lens, cum_seq_lens = generate_varlen_inputs(
            BATCH_SIZE, SEQ_LEN
        )
        t_expanded = cum_seq_lens[-1]
        t_compact = t_expanded
    setup_varlen_attention(model, cum_seq_lens, 'npu')

    print(f"seq_lens: {seq_lens[:5]}, cum_seq_lens: {cum_seq_lens}")

    # 2.1 预计算 max_seq_len 的 cos/sin 表, 按 expanded position_ids gather
    precompute_rope_cos_sin(model, MAX_SEQ_LEN, 'npu')
    cos_table = model.model.rotary_emb._cached_cos  # [1, MAX_SEQ_LEN, 64]
    sin_table = model.model.rotary_emb._cached_sin
    pos = expanded_pos.squeeze(0).npu()  # [T_e] = [0..S-1, 0..S-1, ...]
    model.model.rotary_emb._cached_cos = cos_table[:, pos, :]  # [1, T_e, 64]
    model.model.rotary_emb._cached_sin = sin_table[:, pos, :]
    print(f"cos/sin gathered: table=[1,{MAX_SEQ_LEN},64] → pos={pos.shape} → cos={model.model.rotary_emb._cached_cos.shape}")

    # 3. 构建 prefix sharing indices
    if prefix_len > 0:
        expand_index = build_expand_index(prefix_len, seq_lens, 'npu')
        restore_index = build_restore_index(prefix_len, seq_lens, 'npu')
        compact_last_indices = build_compact_last_indices(prefix_len, seq_lens, 'npu')
    else:
        expand_index = torch.arange(t_expanded, dtype=torch.int64, device='npu')
        restore_index = torch.arange(t_compact, dtype=torch.int64, device='npu')
        compact_last_indices = torch.tensor(cum_seq_lens, dtype=torch.int64, device='npu') - 1

    # 4. 生成 golden logits (eager 模式, 复用 ExportWrapper 保证与导出路径一致)
    print("=== 生成 golden logits (仅每条序列最后一个 token) ===")
    asl_tensor = torch.tensor(cum_seq_lens, dtype=torch.int64, device='npu')
    cos = model.model.rotary_emb._cached_cos
    sin = model.model.rotary_emb._cached_sin
    with torch.no_grad():
        golden_logits = ExportWrapper(model)(
            compact_ids.squeeze(0).npu(),
            expanded_pos.squeeze(0).npu(),
            asl_tensor, cos, sin,
            expand_index, restore_index, compact_last_indices,
        ).cpu()
    print(f"golden_logits shape: {golden_logits.shape}")

    # 5. 收集 7 个用户输入 (按 OM Data 节点顺序)
    inputs = [
        ("actual_seq_lengths", torch.tensor(cum_seq_lens, dtype=torch.int64).cpu()),
        ("cos", model.model.rotary_emb._cached_cos.cpu()),
        ("sin", model.model.rotary_emb._cached_sin.cpu()),
        ("input_ids", compact_ids.squeeze(0).cpu()),
        ("expand_index", expand_index.cpu()),
        ("restore_index", restore_index.cpu()),
        ("compact_last_indices", compact_last_indices.cpu()),
    ]

    # 6. 保存
    os.makedirs(args.output_dir, exist_ok=True)

    list_lines = []
    arg_names = ["arg1_1", "arg3_1", "arg5_1", "arg8_1", "arg9_1", "arg10_1", "arg11_1"]
    for idx, (name, tensor) in enumerate(inputs):
        fname = f"{name}.bin"
        fpath = os.path.join(args.output_dir, fname)
        tensor.detach().numpy().tofile(fpath)

        shape = ",".join(str(s) for s in tensor.shape)
        if tensor.dtype == torch.float16:
            dtype = "float16"
        elif tensor.dtype == torch.int64:
            dtype = "int64"
        elif tensor.dtype == torch.bool:
            dtype = "bool"
        else:
            dtype = str(tensor.dtype).replace("torch.", "")

        list_lines.append(f"{arg_names[idx]}:{shape}:{dtype}:ND:{fpath}")
        print(f"  [{idx}] {arg_names[idx]:12s} {name:25s} shape={str(tensor.shape):25s} dtype={dtype}")

    list_path = os.path.join(args.output_dir, "input_list.txt")
    with open(list_path, 'w') as f:
        for line in list_lines:
            f.write(line + "\n")
    print(f"\nInput list saved to: {list_path} ({len(inputs)} inputs)")

    # 7. 保存 golden logits
    golden_path = os.path.join(args.output_dir, "golden_logits.bin")
    golden_logits.detach().numpy().tofile(golden_path)
    print(f"Golden logits saved to: {golden_path}")

    # 8. 保存 vocab map (prune 模式下, 与 export_air.py 一致)
    if token_ids:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        token_names = [tokenizer.decode([tid]) for tid in token_ids]
        map_path = os.path.join(args.output_dir, "golden_vocab_map.json")
        with open(map_path, "w") as f:
            json.dump({
                "original_token_ids": token_ids,
                "token_names": token_names,
                "pruned_vocab_size": len(token_ids),
            }, f, indent=2, ensure_ascii=False)
        print(f"Vocab map saved to: {map_path}")


if __name__ == "__main__":
    main()
