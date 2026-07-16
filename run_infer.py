"""
主入口 — Qwen2.5 varlen NPU 推理 (npu_fused_infer_attention_score + torchair dynamic=True)

支持三种执行模式:
    eager:       PyTorch eager 模式, 无图编译, 无融合算子
    graph:       torchair GE 图编译, 无融合算子
    graph_fused: torchair GE 图编译 + NPU 融合算子

用法:
    python -m qwen_varlen.run_infer
    python -m qwen_varlen.run_infer --mode eager --profiling
    python -m qwen_varlen.run_infer --mode graph --profiling
    python -m qwen_varlen.run_infer --mode graph_fused --profiling
    python -m qwen_varlen.run_infer --mode all --profiling
"""

import argparse

import torch
import torchair
from transformers import AutoModelForCausalLM, AutoTokenizer

from .attention import register_npu_fia
from .varlen_utils import prepare_varlen_inputs, setup_varlen_attention
from .fusion_ops import apply_fusion_ops
from .export_air import patch_attention_for_dynamic
from .profiling import run_with_profiling

DEFAULT_MODEL_PATH = "/export/home/models/Qwen2.5-0.5B"
DEFAULT_INPUT_TEXTS = ["你好", "请介绍一下自己", "写一段Python快速排序代码"]
DEFAULT_DEVICE = 8

LONG_TEXT_TEMPLATE = (
    "请详细介绍人工智能的发展历史，从图灵测试开始，到深度学习的兴起，"
    "再到大语言模型的爆发。重点关注每个阶段的关键技术突破和代表性工作。"
)

VALID_MODES = ["eager", "graph", "graph_fused", "all"]


def generate_input_texts(batch_size, seq_len):
    """生成指定 batch_size 和近似 seq_len 的输入文本列表。"""
    if batch_size <= 0:
        return DEFAULT_INPUT_TEXTS
    base = LONG_TEXT_TEMPLATE
    base_tokens = len(base) // 2
    repeat = max(1, seq_len // base_tokens)
    text = (base * repeat)[:seq_len * 2]
    return [text] * batch_size


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen2.5 varlen NPU 推理")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="模型路径")
    parser.add_argument("--device", type=int, default=DEFAULT_DEVICE, help="NPU 设备号")
    parser.add_argument("--batch-size", type=int, default=10, help="batch size (0=用默认短文本)")
    parser.add_argument("--seq-len", type=int, default=208, help="每条文本近似 token 数")
    parser.add_argument("--mode", choices=VALID_MODES, default="graph_fused",
                        help="执行模式: eager | graph | graph_fused | all")
    parser.add_argument("--profiling", action="store_true", help="是否采集 profiling")
    parser.add_argument("--prof-dir", default="./log", help="profiling 输出根目录")
    parser.add_argument("--prof-iters", type=int, default=100, help="profiling 采集轮数")
    parser.add_argument("--logits-to-keep", type=int, default=1,
                        help="只保留最后 N 个 token 的 logits (0=全部, 1=仅最后1个, 适合搜索相关性)")
    return parser.parse_args()


def load_model(model_path, device, use_fusion):
    """加载模型到指定 NPU, 注册自定义 attention, 可选应用融合算子。"""
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
        patch_attention_for_dynamic()

    return model


def compile_model(model):
    """torchair GE 图模式编译, dynamic=True 支持多 shape 复用。"""
    config = torchair.CompilerConfig()
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    return torch.compile(model, backend=npu_backend, dynamic=True)


def print_predictions(logits, seq_lens, input_texts, tokenizer):
    """打印预测结果。

    logits_to_keep=1 时 logits shape 为 [1, 1, vocab], 只输出最后 1 个 token。
    logits_to_keep=0 时 logits shape 为 [1, total_len, vocab], 按 seq_lens 拆分。
    """
    print(f"  logits shape: {logits.shape}")
    if logits.shape[1] == 1:
        pred = tokenizer.decode(logits[0, -1].argmax())
        print(f"  pred (last token): {pred}")
    else:
        splits = torch.split(logits[0], seq_lens, dim=0)
        num_to_print = min(3, len(input_texts))
        for i in range(num_to_print):
            text = input_texts[i]
            split = splits[i]
            preview = text[:40].replace('\n', ' ') + "..." if len(text) > 40 else text
            print(f"  Seq {i+1} '{preview}': pred={tokenizer.decode(split[-1].argmax())}")
        if len(input_texts) > num_to_print:
            print(f"  ... ({len(input_texts)} sequences total, showing first {num_to_print})")


def run_single_mode(mode, args, tokenizer, input_texts):
    """运行单个模式的推理或 profiling。

    Args:
        mode: "eager" | "graph" | "graph_fused"
        args: CLI 参数
        tokenizer: tokenizer 实例
        input_texts: 输入文本列表
    """
    print(f"\n{'='*60}")
    print(f"  MODE: {mode}")
    print(f"{'='*60}")

    use_fusion = (mode == "graph_fused")
    use_compile = (mode != "eager")

    # 1. 加载模型 (每次重新加载, 避免 monkey-patch 互相污染)
    model = load_model(args.model_path, args.device, use_fusion=use_fusion)

    # 2. 准备 varlen 输入
    concat_ids, concat_pos, seq_lens, cum_seq_lens = prepare_varlen_inputs(tokenizer, input_texts)
    setup_varlen_attention(model, cum_seq_lens, 'npu')
    inputs = {
        "input_ids": concat_ids.npu(),
        "position_ids": concat_pos.npu(),
        "attention_mask": torch.ones_like(concat_ids).npu(),
        "logits_to_keep": args.logits_to_keep,
    }

    print(f"  batch_size={len(input_texts)}, total_tokens={sum(seq_lens)}, "
          f"seq_lens[:5]={seq_lens[:5]}, cum_seq_lens[-1]={cum_seq_lens[-1]}")

    # 3. 编译 (eager 模式跳过)
    if use_compile:
        print(f"=== TorchAir GE 图模式 (dynamic=True) ===")
        model_fn = compile_model(model)
    else:
        print(f"=== Eager 模式 (无图编译) ===")
        model_fn = model

    # 4. 推理 / profiling
    prof_dir = f"{args.prof_dir}/{mode}"

    if args.profiling:
        logits = run_with_profiling(model_fn, inputs, prof_dir, num_iters=args.prof_iters)
    else:
        print("=== Warmup ===")
        with torch.no_grad():
            _ = model_fn(**inputs)
        torch.npu.synchronize()
        print("=== 正式推理 ===")
        with torch.no_grad():
            logits = model_fn(**inputs).logits
        torch.npu.synchronize()

    # 5. 打印结果
    print_predictions(logits, seq_lens, input_texts, tokenizer)

    # 6. 清理 NPU 缓存 (避免多模式间内存累积)
    torch.npu.empty_cache()


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    input_texts = generate_input_texts(args.batch_size, args.seq_len)

    if args.mode == "all":
        for mode in ["eager", "graph", "graph_fused"]:
            run_single_mode(mode, args, tokenizer, input_texts)
    else:
        run_single_mode(args.mode, args, tokenizer, input_texts)


if __name__ == "__main__":
    main()
