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
from .varlen_utils import build_expand_index, build_restore_index, build_compact_last_indices
from .fusion_ops import apply_fusion_ops
from atb.tools.varlen import generate_varlen_inputs, generate_compact_varlen_inputs
from atb.tools.atc_utils import run_atc
from atb.tools.lm_head_prune import load_target_tokens, prune_lm_head


def _patched_attention_forward(self, hidden_states, position_embeddings,
                               attention_mask, past_key_values=None, **kwargs):
    """Qwen2Attention.forward 的动态导出兼容版本 (2D 模式 + prefix sharing)。

    hidden_states: [T_c, hidden_size] (2D, compact layout)
    所有 reshape 仅使用 Python int 常量 + -1, 不提取 SymInt, 消除 GE Pack 算子。

    prefix sharing: expand_index/restore_index 由 ExportWrapper.forward 注入到 self。
    - QKV 前 expand: compact [T_c] → expanded [T_e]
    - attention 后 restore: expanded [T_e] → compact [T_c]
    - o_proj 跑 compact, 省计算
    """

    num_heads = int(self.config.num_attention_heads)
    num_kv_heads = int(self.config.num_key_value_heads)
    hidden_size = int(self.config.hidden_size)
    head_dim = int(self.head_dim)

    expand_index = getattr(self, 'expand_index', None)
    if expand_index is not None:
        hidden_states = torch.index_select(hidden_states, 0, expand_index)

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

    restore_index = getattr(self, 'restore_index', None)
    if restore_index is not None:
        attn_output = torch.index_select(attn_output, 0, restore_index)

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

    prefix sharing 模式 (prefix_len > 0):
        input_ids 为 compact [prefix, req_1, ..., req_n], 省存储。
        attention 内部 expand→QKV→Attn→restore, o_proj 跑 compact。

    动态输入 (forward 参数, 成为图 Data 节点):
        input_ids:            [T_c] int64 — compact token ids (T_c 动态)
        position_ids:         [T_e] int64 — expanded position ids (图中消除)
        actual_seq_lengths:   [num_batch] int64 — expanded 累积序列长度
        cos:                  [1, T_e, 64] float16 — RoPE cos (expanded)
        sin:                  [1, T_e, 64] float16 — RoPE sin (expanded)
        expand_index:         [T_e] int64 — compact→expanded 映射
        restore_index:        [T_c] int64 — expanded→compact 映射
        compact_last_indices: [num_batch] int64 — 每条序列最后 token 在 compact 中的位置

    输出:
        logits: [N, vocab_size] float16 — 仅每条序列最后一个 token 的 logits
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, position_ids, actual_seq_lengths, cos, sin,
                expand_index, restore_index, compact_last_indices):
        for layer in self.model.model.layers:
            layer.self_attn.actual_seq_lengths_tensor = actual_seq_lengths
        self.model.model.rotary_emb._cached_cos = cos
        self.model.model.rotary_emb._cached_sin = sin

        m = self.model.model
        hidden = m.embed_tokens(input_ids)

        for layer in m.layers:
            layer.self_attn.expand_index = expand_index
            layer.self_attn.restore_index = restore_index

        position_embeddings = m.rotary_emb(hidden, position_ids)
        for layer in m.layers:
            hidden = layer(
                hidden,
                attention_mask=None,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                use_cache=False,
            )
        last_hidden = hidden.index_select(0, compact_last_indices)
        last_hidden = m.norm(last_hidden)
        return self.model.lm_head(last_hidden)


def export_air(model_path, output_dir, device, batch_size, seq_len, prefix_len=0,
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
        seq_len:           每条文本 token 数 (含 prefix)
        prefix_len:        共享前缀长度 (0 则不启用 prefix sharing)
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

    # 3. 准备 varlen 输入
    if prefix_len > 0:
        compact_ids, expanded_pos, seq_lens, cum_seq_lens = generate_compact_varlen_inputs(
            batch_size, seq_len, prefix_len
        )
        t_expanded = cum_seq_lens[-1]
        t_compact = prefix_len + batch_size * (seq_len - prefix_len)
        print(f"  prefix sharing: prefix_len={prefix_len}, T_compact={t_compact}, T_expanded={t_expanded}")
    else:
        compact_ids, expanded_pos, seq_lens, cum_seq_lens = generate_varlen_inputs(
            batch_size, seq_len
        )
        t_expanded = cum_seq_lens[-1]
        t_compact = t_expanded
    setup_varlen_attention(model, cum_seq_lens, 'npu')

    print(f"  batch_size={batch_size}, T_expanded={t_expanded}, "
          f"seq_lens[:5]={seq_lens[:5]}, cum_seq_lens[-1]={cum_seq_lens[-1]}")

    # 4. 构造 dummy 输入 (NPU 上)
    input_ids = compact_ids.squeeze(0).npu()
    position_ids = expanded_pos.squeeze(0).npu()
    actual_seq_lengths = torch.tensor(cum_seq_lens, dtype=torch.int64, device='npu')
    cos = model.model.rotary_emb._cached_cos
    sin = model.model.rotary_emb._cached_sin

    # 4.1 prefix sharing indices
    if prefix_len > 0:
        expand_index = build_expand_index(prefix_len, seq_lens, 'npu')
        restore_index = build_restore_index(prefix_len, seq_lens, 'npu')
        compact_last_indices = build_compact_last_indices(prefix_len, seq_lens, 'npu')
    else:
        expand_index = torch.arange(t_expanded, dtype=torch.int64, device='npu')
        restore_index = torch.arange(t_compact, dtype=torch.int64, device='npu')
        compact_last_indices = actual_seq_lengths - 1

    # 4.2 精确标记动态/静态维度
    torch._dynamo.mark_dynamic(input_ids, 0)            # [T_c] → T_c 动态
    torch._dynamo.mark_dynamic(position_ids, 0)         # [T_e] → T_e 动态 (图中消除)
    torch._dynamo.mark_dynamic(actual_seq_lengths, 0)   # [N] → N 动态
    torch._dynamo.mark_dynamic(cos, 1)                  # [1, T_e, 64] → T_e 动态
    torch._dynamo.mark_dynamic(sin, 1)                  # [1, T_e, 64] → T_e 动态
    torch._dynamo.mark_static(cos, 0)
    torch._dynamo.mark_static(cos, 2)
    torch._dynamo.mark_static(sin, 0)
    torch._dynamo.mark_static(sin, 2)
    torch._dynamo.mark_dynamic(expand_index, 0)         # [T_e] → T_e 动态
    torch._dynamo.mark_dynamic(restore_index, 0)        # [T_c] → T_c 动态
    torch._dynamo.mark_dynamic(compact_last_indices, 0) # [N] → N 动态

    # 5. 包装模型
    export_model = ExportWrapper(model)

    # 6. 配置 CompilerConfig (frozen_parameter 配合动态导出)
    config = CompilerConfig()
    config.experimental_config.frozen_parameter = 1

    # 7. 导出 AIR
    os.makedirs(output_dir, exist_ok=True)

    print(f"=== 导出 AIR (动态): {output_dir}/{export_name}.air ===")
    print(f"  input_ids: {input_ids.shape}, actual_seq_lengths: {actual_seq_lengths.shape}")
    print(f"  cos: {cos.shape}, sin: {sin.shape}")
    print(f"  expand_index: {expand_index.shape}, restore_index: {restore_index.shape}")
    print(f"  compact_last_indices: {compact_last_indices.shape}")

    dynamo_export(
        input_ids, position_ids, actual_seq_lengths, cos, sin,
        expand_index, restore_index, compact_last_indices,
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
    parser.add_argument("--seq-len", type=int, default=208, help="每条文本 token 数 (含 prefix)")
    parser.add_argument("--prefix-len", type=int, default=0, help="共享前缀长度 (0 则不启用 prefix sharing)")
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
            prefix_len=args.prefix_len,
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
