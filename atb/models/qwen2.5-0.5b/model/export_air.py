"""
导出 AIR 模型 — PyTorch(NPU融合算子) → torchair.dynamo_export → AIR → ATC → OM

导出链路: PyTorch → AIR → OM
保留 NPU 融合算子 (FusedInferAttentionScore, RmsNorm, RotaryMul), FFN/QKV 保持小算子。

用法:
    python -m model.export_air --device 0
    python -m model.export_air --device 0 --run-atc --soc Ascend910_9382
"""

import os
import json
import logging
import argparse

import torch
import torch.nn as nn
import torch_npu
from torch_npu.dynamo.torchair import dynamo_export, CompilerConfig
from transformers import AutoModelForCausalLM
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen2 import modeling_qwen2

from .attention import register_npu_fia
from .varlen_utils import setup_varlen_attention
from .fusion_ops import apply_fusion_ops
from atb.tools.varlen import generate_varlen_inputs
from atb.tools.atc_utils import run_atc
from atb.tools.lm_head_prune import load_target_tokens, prune_lm_head


def _patched_attention_forward(self, hidden_states, position_embeddings,
                               attention_mask, past_key_values=None, **kwargs):
    """Qwen2Attention.forward 的动态导出兼容版本 (2D 模式)。

    hidden_states: [T, hidden_size] (2D, 无 batch 维度)
    所有 reshape 仅使用 Python int 常量 + -1, 不提取 SymInt, 消除 GE Pack 算子。
    """

    num_heads = int(self.config.num_attention_heads)
    num_kv_heads = int(self.config.num_key_value_heads)
    hidden_size = int(self.config.hidden_size)
    head_dim = int(self.head_dim)

    query_states = self.q_proj(hidden_states).reshape(-1, num_heads, head_dim)
    key_states = self.k_proj(hidden_states).reshape(-1, num_kv_heads, head_dim)
    value_states = self.v_proj(hidden_states).reshape(-1, num_kv_heads, head_dim)

    cos, sin = position_embeddings
    query_states, key_states = modeling_qwen2.apply_rotary_pos_emb(query_states, key_states, cos, sin)

    attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation, modeling_qwen2.eager_attention_forward
    )
    attn_output, attn_weights = attention_interface(
        self, query_states, key_states, value_states, attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling, sliding_window=self.sliding_window, **kwargs,
    )

    attn_output = attn_output.reshape(-1, hidden_size).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def patch_attention_for_dynamic():
    """Monkey-patch Qwen2Attention.forward, 2D 模式避免 Pack。"""
    modeling_qwen2.Qwen2Attention.forward = _patched_attention_forward
    print("[patch] Qwen2Attention.forward → 动态导出版 (2D 模式, reshape 用 -1 避免 Pack)")


def load_model(model_path, device):
    """加载模型 (NPU, npu_fia, fp16), 应用融合算子 + attention patch。

    export_air 和 prepare_air_inputs 共用此函数, 保证两条路径模型状态一致。
    """
    torch.npu.set_device(device)
    register_npu_fia()
    print(f"=== 加载模型 (attn_implementation='npu_fia', device={device}) ===")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.float16, attn_implementation="npu_fia"
    ).npu()
    model.eval()
    model.config.use_cache = False
    print("=== 应用融合算子 ===")
    apply_fusion_ops()
    patch_attention_for_dynamic()
    return model


_MODEL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL_PATH = "/export/home/models/Qwen2.5-0.5B"
DEFAULT_MODEL_NAME = "qwen2.5-0.5b"
DEFAULT_OUTPUT_DIR = os.path.join(_MODEL_DIR, "air")
DEFAULT_OM_DIR = os.path.join(_MODEL_DIR, "om")
DEFAULT_TARGET_TOKEN_FILE = os.path.join(_MODEL_DIR, "target_tokens.json")
DEFAULT_SOC = "Ascend910_9382"
DEFAULT_DEVICE = 8


class ExportWrapper(nn.Module):
    """包装模型, 2D 格式输入, 完全动态导出。

    核心设计: hidden_states 全程保持 2D [T, D], 不添加 batch 维度,
    避免 nn.Linear 内部 reshape back 产生 GE Pack 算子。

    动态输入 (forward 参数, 成为图 Data 节点):
        input_ids:            [T] int64 — 所有 token 拼接 (T 动态)
        position_ids:         [T] int64 — 对应 position ids (T 动态, 图中消除)
        actual_seq_lengths:   [num_batch] int64 — 累积序列长度 (num_batch 动态)
        cos:                  [1, T, 64] float16 — RoPE cos (T 动态)
        sin:                  [1, T, 64] float16 — RoPE sin (T 动态)

    固定输入 (module 属性, 成为图 Data 节点但 shape 固定):
        atten_mask:            [2048, 2048] bool — 因果掩码 (固定)

    图常量 (register_buffer, 不成为图输入):
        FFN 权重 (_ffn_w1, _ffn_w2), 模型权重 (embeddings, q/k/v/o proj, lm_head)

    输出:
        logits: [N, vocab_size] float16 — 仅每条序列最后一个 token 的 logits
        N = num_batch (固定), 搜索相关性只需最后一个 token 做分类
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, position_ids, actual_seq_lengths, cos, sin):
        for layer in self.model.model.layers:
            layer.self_attn.actual_seq_lengths_tensor = actual_seq_lengths
        self.model.model.rotary_emb._cached_cos = cos
        self.model.model.rotary_emb._cached_sin = sin

        m = self.model.model
        hidden = m.embed_tokens(input_ids)
        position_embeddings = m.rotary_emb(hidden, position_ids)
        for layer in m.layers:
            hidden = layer(
                hidden,
                attention_mask=None,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                use_cache=False,
            )
        last_indices = actual_seq_lengths - 1
        last_hidden = hidden.index_select(0, last_indices)
        last_hidden = m.norm(last_hidden)
        return self.model.lm_head(last_hidden)


def export_air(model_path, output_dir, device, batch_size, seq_len,
               export_name="qwen2.5-0.5b", prune=False, target_token_file=None):
    """导出 AIR 模型。

    流程:
      1. 加载模型 (NPU, npu_fia, fp16) — 与 graph_fused 推理模式一致
      2. 应用融合算子 (RMSNorm + RoPE)
      2.1 lm_head vocab 剪裁 (可选)
      3. 设置 varlen 参数 (actual_seq_lengths, atten_mask, cos/sin 预计算)
      4. 包装模型 (ExportWrapper, 只返回 logits)
      5. dynamo_export 导出 AIR (动态 shape)

    Args:
        model_path:        模型路径
        output_dir:        AIR 输出目录
        device:            NPU 设备号
        batch_size:        batch size
        seq_len:           每条文本近似 token 数
        prune:             是否开启 lm_head vocab 剪裁
        target_token_file: target token JSON 文件路径 (prune=True 时必填)
    """
    logging.getLogger('torchair').setLevel(logging.INFO)

    # 1. 加载模型 (NPU, npu_fia, fp16)
    model = load_model(model_path, device)

    # 2. lm_head vocab 剪裁
    token_ids = None
    if prune:
        if not target_token_file or not os.path.exists(target_token_file):
            raise FileNotFoundError(f"target_token_file 不存在: {target_token_file}")
        token_ids = load_target_tokens(target_token_file)
        prune_lm_head(model, token_ids)

    # 3. 准备 varlen 输入 (全 0 token, 不需要 tokenizer)
    concat_ids, concat_pos, seq_lens, cum_seq_lens = generate_varlen_inputs(
        batch_size, seq_len
    )
    setup_varlen_attention(model, cum_seq_lens, 'npu')

    print(f"  batch_size={batch_size}, total_tokens={sum(seq_lens)}, "
          f"seq_lens[:5]={seq_lens[:5]}, cum_seq_lens[-1]={cum_seq_lens[-1]}")

    # 4. 构造 dummy 输入 (NPU 上, TND 格式)
    input_ids = concat_ids.squeeze(0).npu()
    position_ids = concat_pos.squeeze(0).npu()
    actual_seq_lengths = torch.tensor(cum_seq_lens, dtype=torch.int64, device='npu')
    cos = model.model.rotary_emb._cached_cos
    sin = model.model.rotary_emb._cached_sin

    # 4.1 精确标记动态/静态维度
    #    T (total tokens) 动态, N (序列数) 固定, D (head_dim) 固定
    torch._dynamo.mark_dynamic(input_ids, 0)            # [T] → T 动态
    torch._dynamo.mark_dynamic(position_ids, 0)         # [T] → T 动态
    torch._dynamo.mark_dynamic(actual_seq_lengths, 0)   # [N] → N 动态 (batch size 可变)
    torch._dynamo.mark_dynamic(cos, 1)                  # [1, T, 64] → T 动态
    torch._dynamo.mark_dynamic(sin, 1)                  # [1, T, 64] → T 动态
    torch._dynamo.mark_static(cos, 0)                   # [1, T, 64] → B=1 固定
    torch._dynamo.mark_static(cos, 2)                   # [1, T, 64] → D=64 固定
    torch._dynamo.mark_static(sin, 0)                   # [1, T, 64] → B=1 固定
    torch._dynamo.mark_static(sin, 2)                   # [1, T, 64] → D=64 固定

    # 5. 包装模型
    export_model = ExportWrapper(model)

    # 6. 配置 CompilerConfig (frozen_parameter 配合动态导出)
    config = CompilerConfig()
    config.experimental_config.frozen_parameter = 1

    # 7. 导出 AIR
    os.makedirs(output_dir, exist_ok=True)

    print(f"=== 导出 AIR (动态): {output_dir}/{export_name}.air ===")
    print(f"  input_ids: {input_ids.shape}, position_ids: {position_ids.shape}")
    print(f"  actual_seq_lengths: {actual_seq_lengths.shape}")
    print(f"  cos: {cos.shape}, sin: {sin.shape}")

    dynamo_export(
        input_ids, position_ids, actual_seq_lengths, cos, sin,
        model=export_model,
        export_path=output_dir,
        export_name=export_name,
        dynamic=True,
        config=config,
    )

    torch.npu.synchronize()

    air_path = os.path.join(output_dir, f"{export_name}.air")
    pbtxt_path = os.path.join(output_dir, "dynamo.pbtxt")

    if os.path.exists(air_path):
        file_size = os.path.getsize(air_path) / 1024 / 1024
        print(f"=== AIR 导出完成: {air_path} ({file_size:.1f} MB) ===\n")
    else:
        print(f"=== [WARN] AIR 文件未生成: {air_path} ===")
        if os.path.exists(pbtxt_path):
            print(f"  dynamo.pbtxt 已生成: {pbtxt_path}")
        print("  请检查上方日志中的 'export error!' 信息\n")

    if token_ids:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        token_names = [tokenizer.decode([tid]) for tid in token_ids]
        map_path = os.path.join(output_dir, f"{export_name}_vocab_map.json")
        with open(map_path, "w") as f:
            json.dump({
                "original_token_ids": token_ids,
                "token_names": token_names,
                "pruned_vocab_size": len(token_ids),
            }, f, indent=2, ensure_ascii=False)
        print(f"=== vocab map 已保存: {map_path} ===\n")

    return air_path


def main():
    parser = argparse.ArgumentParser(description="导出 AIR 模型 (torchair.dynamo_export)")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="模型路径")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="导出模型名称 (AIR/OM 文件名)")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="AIR 输出目录")
    parser.add_argument("--om-dir", default=DEFAULT_OM_DIR, help="OM 输出目录")
    parser.add_argument("--device", type=int, default=DEFAULT_DEVICE, help="NPU 设备号")
    parser.add_argument("--batch-size", type=int, default=10, help="batch size")
    parser.add_argument("--seq-len", type=int, default=208, help="每条文本 token 数")
    parser.add_argument("--soc", default=DEFAULT_SOC, help="SoC 型号")
    parser.add_argument("--run-atc", action="store_true", help="自动执行 ATC 编译")
    parser.add_argument("--skip-export", action="store_true",
                        help="跳过导出, 直接用已有 AIR 做 ATC 编译")
    parser.add_argument("--debug", action="store_true", help="ATC 编译时开启 --log=debug")
    parser.add_argument("--aicore-num", default=None,
                        help="ATC 编译核数: 传单个整数 N 视为 AIC 核数, AIV=N*2 (如 12→12|24); "
                             "传 'aic|aiv' 原样透传; 不传则默认全核")
    parser.add_argument("--prune-lm-head", action="store_true",
                        help="开启 lm_head vocab 剪裁")
    parser.add_argument("--target-token-file", default=DEFAULT_TARGET_TOKEN_FILE,
                        help=f'target token JSON 文件 (默认: {DEFAULT_TARGET_TOKEN_FILE})')
    args = parser.parse_args()

    air_path = os.path.join(args.output_dir, f"{args.model_name}.air")
    if args.skip_export:
        if not os.path.exists(air_path):
            print(f"[ERROR] AIR 文件不存在: {air_path}, 请先去掉 --skip-export 导出")
            return
        print(f"=== 跳过导出, 使用已有 AIR: {air_path} ===")
    else:
        air_path = export_air(
            args.model_path,
            args.output_dir,
            args.device,
            args.batch_size,
            args.seq_len,
            export_name=args.model_name,
            prune=args.prune_lm_head,
            target_token_file=args.target_token_file,
        )

    if args.run_atc:
        om_path = run_atc(air_path, args.om_dir, args.soc,
                          is_debug=args.debug, aicore_num=args.aicore_num)
        if om_path:
            print(f"=== 全流程完成 ===")
            print(f"  AIR: {air_path}")
            print(f"  OM:  {om_path}")


if __name__ == "__main__":
    main()
