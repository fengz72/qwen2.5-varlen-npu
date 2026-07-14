"""
Qwen2.5 varlen 推理 — npu_fused_infer_attention_score (推理算子) + dynamic=True

模块结构:
    attention.py    — 自定义 NPU 推理 attention 函数 + 注册
    varlen_utils.py — 变长输入处理 (token 拼接, position ids, mask 注入)
    fusion_ops.py   — 融合算子替换 (RMSNorm/RoPE/SwiGLU → NPU 融合算子)
    profiling.py    — Profiling 采集封装 (warmup + 正式采集)
    run_infer.py    — 主入口 (模型加载 → 编译 → 推理/profiling)
"""

from .attention import register_npu_fia, build_causal_mask_2048
from .varlen_utils import prepare_varlen_inputs, setup_varlen_attention
from .fusion_ops import apply_fusion_ops
from .profiling import run_with_profiling
