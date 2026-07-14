"""
主入口 — Qwen2.5 varlen 推理 (npu_fused_infer_attention_score + torchair dynamic=True)

用法:
    python -m qwen_varlen.run_infer
    python -m qwen_varlen.run_infer --model-path /path/to/model --profiling
"""

import argparse

import torch
import torchair
from transformers import AutoModelForCausalLM, AutoTokenizer

from .attention import register_npu_fia
from .varlen_utils import prepare_varlen_inputs, setup_varlen_attention
from .fusion_ops import apply_fusion_ops
from .profiling import run_with_profiling

DEFAULT_MODEL_PATH = "/export/home/models/Qwen2.5-0.5B"
DEFAULT_INPUT_TEXTS = ["你好", "请介绍一下自己", "写一段Python快速排序代码"]
DEFAULT_DEVICE = 8

LONG_TEXT_TEMPLATE = (
    "请详细介绍人工智能的发展历史，从图灵测试开始，到深度学习的兴起，"
    "再到大语言模型的爆发。重点关注每个阶段的关键技术突破和代表性工作。"
)


def generate_input_texts(batch_size, seq_len):
    """生成指定 batch_size 和近似 seq_len 的输入文本列表。"""
    if batch_size <= 0:
        return DEFAULT_INPUT_TEXTS
    base = LONG_TEXT_TEMPLATE
    base_tokens = len(base) // 2  # 近似 token 数
    repeat = max(1, seq_len // base_tokens)
    text = (base * repeat)[:seq_len * 2]
    return [text] * batch_size


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen2.5 varlen NPU 推理")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="模型路径")
    parser.add_argument("--device", type=int, default=DEFAULT_DEVICE, help="NPU 设备号")
    parser.add_argument("--batch-size", type=int, default=10, help="batch size (0=用默认短文本)")
    parser.add_argument("--seq-len", type=int, default=208, help="每条文本近似 token 数")
    parser.add_argument("--profiling", action="store_true", help="是否采集 profiling")
    parser.add_argument("--prof-dir", default="./prof_output", help="profiling 输出目录")
    parser.add_argument("--no-fusion", action="store_true", help="禁用融合算子")
    return parser.parse_args()


def load_model(model_path, device, use_fusion=True):
    """加载模型到指定 NPU, 注册自定义 attention, 应用融合算子。"""
    register_npu_fia()
    torch.npu.set_device(device)
    print(f"=== 加载模型 (attn_implementation='npu_fia', device={device}) ===")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, attn_implementation="npu_fia"
    ).npu()
    model.eval()
    model.config.use_cache = False

    if use_fusion:
        print("=== 应用融合算子 ===")
        apply_fusion_ops(model)

    return model


def compile_model(model):
    """torchair GE 图模式编译, dynamic=True 支持多 shape 复用。"""
    config = torchair.CompilerConfig()
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    return torch.compile(model, backend=npu_backend, dynamic=True)


def print_predictions(logits, seq_lens, input_texts, tokenizer):
    """按 seq_lens 拆分 logits, 打印每条文本的预测。"""
    print(f"  [Graph] logits shape: {logits.shape}")
    splits = torch.split(logits[0], seq_lens, dim=0)
    num_to_print = min(3, len(input_texts))
    for i in range(num_to_print):
        text = input_texts[i]
        split = splits[i]
        preview = text[:40].replace('\n', ' ') + "..." if len(text) > 40 else text
        print(f"  Seq {i+1} '{preview}': pred={tokenizer.decode(split[-1].argmax())}")
    if len(input_texts) > num_to_print:
        print(f"  ... ({len(input_texts)} sequences total, showing first {num_to_print})")


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    input_texts = generate_input_texts(args.batch_size, args.seq_len)

    # 1. 加载模型
    model = load_model(args.model_path, args.device, use_fusion=not args.no_fusion)

    # 2. 准备 varlen 输入
    concat_ids, concat_pos, seq_lens, cum_seq_lens = prepare_varlen_inputs(tokenizer, input_texts)
    setup_varlen_attention(model, cum_seq_lens, 'npu')
    inputs = {
        "input_ids": concat_ids.npu(),
        "position_ids": concat_pos.npu(),
        "attention_mask": torch.ones_like(concat_ids).npu(),
    }

    print(f"\n=== TorchAir 图模式 (GE + npu_fused_infer_attention_score TND, dynamic=True) ===")
    print(f"  batch_size={len(input_texts)}, total_tokens={sum(seq_lens)}, "
          f"seq_lens[:5]={seq_lens[:5]}, cum_seq_lens[-1]={cum_seq_lens[-1]}")

    # 3. 编译模型
    compiled_model = compile_model(model)

    # 4. 推理 (带/不带 profiling)
    if args.profiling:
        logits = run_with_profiling(compiled_model, inputs, args.prof_dir)
    else:
        print("=== Warmup ===")
        with torch.no_grad():
            _ = compiled_model(**inputs)
        torch.npu.synchronize()
        print("=== 正式推理 ===")
        with torch.no_grad():
            logits = compiled_model(**inputs).logits
        torch.npu.synchronize()

    # 5. 打印结果
    print_predictions(logits, seq_lens, input_texts, tokenizer)


if __name__ == "__main__":
    main()
