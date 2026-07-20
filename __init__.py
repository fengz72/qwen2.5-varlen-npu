"""
Qwen2.5 varlen 搜索相关性推理 — PyTorch(NPU融合算子) → AIR → OM

模块结构:
    attention.py          — 自定义 NPU 推理 attention 函数 + 注册
    varlen_utils.py       — 变长输入处理 (token 拼接, position ids, mask 注入)
    fusion_ops.py         — 融合算子替换 (RMSNorm/RoPE/SwiGLU → NPU 融合算子)
    export_air.py         — AIR 导出 (torchair.dynamo_export) + ATC 编译
    prepare_air_inputs.py — 生成 OM 推理输入数据 + golden logits
"""

from .attention import register_npu_fia, build_causal_mask_2048
from .varlen_utils import generate_varlen_inputs, setup_varlen_attention
from .fusion_ops import apply_fusion_ops
