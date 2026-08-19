"""
Qwen2.5 varlen 搜索相关性推理 — PyTorch(NPU融合算子) → AIR → OM

模块结构:
    attention.py          — 自定义 NPU 推理 attention 函数 + 注册
    varlen_utils.py       — 变长输入处理 (position ids, mask 注入, cos/sin 表注册为图常量)
    fusion_ops.py         — 融合算子替换 (RMSNorm/RoPE → NPU 融合算子, FFN/QKV 保持小算子)
    export_air.py         — AIR 导出 (torchair.dynamo_export) + ATC 编译
    prepare_air_inputs.py — 生成 OM 推理输入数据 + golden logits

通用工具 (atb.tools):
    varlen.py             — token 拼接 (generate/prepare_varlen_inputs)
    atc_utils.py          — ATC 编译 (run_atc)
    lm_head_prune.py      — lm_head vocab 剪裁
"""

from .attention import register_npu_fia, build_causal_mask_2048
from .varlen_utils import setup_varlen_attention
from .fusion_ops import apply_fusion_ops
