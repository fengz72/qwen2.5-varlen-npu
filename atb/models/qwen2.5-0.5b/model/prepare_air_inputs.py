"""
为 AIR 导出的 OM 模型生成 3 个用户输入数据 + golden logits。

frozen_parameter=1 后, FFN 权重、atten_mask、cos/sin 表 已冻结为图常量,
OM 只有 3 个图输入 (Data 节点):
  - actual_seq_lengths [N] int64
  - position_ids [T] int64 (用于图内 cos/sin Gather)
  - input_ids [T] int64

cos/sin 表 [1, 1024, 64] 作为 register_buffer 注册, 配合 frozen_parameter=1
成为图常量。图内按 position_ids gather 出 [1, T, 64], 无需用户提供 cos/sin。

同时运行 eager 模式生成 golden logits 供精度对比。
"""

import argparse
import json
import os
import torch
import torch_npu

from .varlen_utils import setup_varlen_attention
from .export_air import load_model, ExportWrapper
from atb.tools.varlen import generate_varlen_inputs
from atb.tools.lm_head_prune import load_target_tokens, prune_lm_head

_MODEL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL_PATH = "/export/home/models/Qwen2.5-0.5B"
DEFAULT_OUTPUT_DIR = os.path.join(_MODEL_DIR, "input_data")
DEFAULT_TARGET_TOKEN_FILE = os.path.join(_MODEL_DIR, "target_tokens.json")
DEFAULT_DEVICE = 0

BATCH_SIZE = 10
SEQ_LEN = 208
MAX_SEQ_LEN = 1024


def main():
    parser = argparse.ArgumentParser(description="生成 OM 输入数据 + golden logits")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="模型路径")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="输出目录")
    parser.add_argument("--device", type=int, default=DEFAULT_DEVICE, help="NPU 设备号")
    parser.add_argument("--prune-lm-head", action="store_true",
                        help="开启 lm_head vocab 剪裁 (与 export_air.py --prune-lm-head 一致)")
    parser.add_argument("--target-token-file", default=DEFAULT_TARGET_TOKEN_FILE,
                        help=f'target token JSON 文件 (默认: {DEFAULT_TARGET_TOKEN_FILE})')
    args = parser.parse_args()

    # 1. 加载模型 (与 export 完全一致)
    model = load_model(args.model_path, args.device)

    # 1.1 lm_head vocab 剪裁 (与 export_air.py 一致)
    token_ids = None
    if args.prune_lm_head:
        if not args.target_token_file or not os.path.exists(args.target_token_file):
            raise FileNotFoundError(f"target_token_file 不存在: {args.target_token_file}")
        token_ids = load_target_tokens(args.target_token_file)
        prune_lm_head(model, token_ids)

    # 2. 准备 varlen 输入 (全 0 token, 不需要 tokenizer)
    concat_ids, concat_pos, seq_lens, cum_seq_lens = generate_varlen_inputs(
        BATCH_SIZE, SEQ_LEN
    )
    setup_varlen_attention(model, cum_seq_lens, 'npu', max_seq_len=MAX_SEQ_LEN)

    print(f"seq_lens: {seq_lens[:5]}, cum_seq_lens: {cum_seq_lens}")

    # 3. 生成 golden logits (eager 模式, 复用 ExportWrapper 保证与导出路径一致)
    print("=== 生成 golden logits (仅每条序列最后一个 token) ===")
    asl_tensor = torch.tensor(cum_seq_lens, dtype=torch.int64, device='npu')
    with torch.no_grad():
        golden_logits = ExportWrapper(model)(
            concat_ids.squeeze(0).npu(),
            concat_pos.squeeze(0).npu(),
            asl_tensor,
        ).cpu()
    print(f"golden_logits shape: {golden_logits.shape}")

    # 4. 收集 3 个用户输入 (arg 名称导出后验证)
    inputs = [
        ("actual_seq_lengths", torch.tensor(cum_seq_lens, dtype=torch.int64).cpu()),
        ("position_ids", concat_pos.squeeze(0).cpu()),
        ("input_ids", concat_ids.squeeze(0).cpu()),
    ]

    # 5. 保存
    os.makedirs(args.output_dir, exist_ok=True)

    list_lines = []
    arg_names = ["arg1_1", "arg3_1", "arg5_1"]
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

    # 6. 保存 golden logits
    golden_path = os.path.join(args.output_dir, "golden_logits.bin")
    golden_logits.detach().numpy().tofile(golden_path)
    print(f"Golden logits saved to: {golden_path}")

    # 7. 保存 vocab map (prune 模式下, 与 export_air.py 一致)
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
