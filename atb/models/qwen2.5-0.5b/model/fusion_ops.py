"""
融合算子替换 — 通过 monkey-patch 将 Qwen2 小算子替换为 NPU 融合算子

替换项:
    1. RMSNorm:  Pow + ReduceMean + Add + Rsqrt + Mul + Cast → npu_rms_norm
    2. RoPE:     rotate_half(StridedSlice + Neg + Cat) + Mul + Add → npu_apply_rotary_pos_emb
    3. RoPE cos/sin: 图内 Cast(206us) + MatMul + Cos + Sin → 图外预计算注入

注意: FFN 不使用 npu_ffn 融合。测试表明 npu_ffn (FFNV3) 的 tiling 策略对
      [2080,896]→[9728]→[4864]→[896] shape 不优, MAC 利用率仅 43.5%,
      拆分为独立 MatMul 后可达 85%, 整体 MLP 提速 16.4%。
      详见 docs/ffn_fusion_optimization.md

注意: QKV 不做融合。测试表明融合后 StridedSliceV3 开销 (1152us/iter)
      远超 MatMul 节省 (243us/iter), 整体慢 8.6%。

用法:
    from model.fusion_ops import apply_fusion_ops
    apply_fusion_ops(model)  # 在模型加载后、编译前调用
"""

import torch
import torch_npu
from transformers.models.qwen2 import modeling_qwen2


# ==================== 1. RMSNorm → npu_rms_norm ====================

def _npu_rms_norm_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    """用 npu_rms_norm 融合算子替代 8 个小算子。

    原始: Cast→Pow→ReduceMean→Add(eps)→Rsqrt→Mul→Cast→Mul(weight)
    融合: npu_rms_norm(x, weight, eps) → 1 个算子
    """
    return torch_npu.npu_rms_norm(hidden_states, self.weight, self.variance_epsilon)[0]


# ==================== 2. RoPE rotate → npu_apply_rotary_pos_emb ====================

def _npu_apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """用 npu_apply_rotary_pos_emb 融合算子替代 2×npu_rotary_mul。

    q/k 为 TND [T, N, D] 格式 (由 _patched_attention_forward 产出)。
    cos/sin 为 [1, T, D], 转换为 [T, 1, D] 供算子使用。
    npu_apply_rotary_pos_emb 一次调用同时处理 Q 和 K, 原生支持 3D TND 布局,
    无需手动 unsqueeze/squeeze 到 4D。

    注意: GE 图编译时 ApplyRotaryPosEmb 可能返回 4D [1,T,N,D] 而非 3D [T,N,D],
    需显式 reshape 回输入 shape, 否则下游 FIA 的 num_heads 推断错误。
    """
    cos = cos.squeeze(0).unsqueeze(1)   # [1, T, D] → [T, 1, D]
    sin = sin.squeeze(0).unsqueeze(1)
    q_embed, k_embed = torch_npu.npu_apply_rotary_pos_emb(
        q, k, cos, sin, layout="TND", rotary_mode="half"
    )
    q_embed = q_embed.reshape(-1, q.size(1), q.size(2))
    k_embed = k_embed.reshape(-1, k.size(1), k.size(2))
    return q_embed, k_embed


# ==================== 3. RoPE cos/sin → 图外预计算 ====================

def _npu_rotary_emb_forward(self, x, position_ids):
    """直接返回图外预计算的 cos/sin, 避免图内 Cast/MatMul/Cos/Sin。

    cos/sin 由 setup_varlen_attention 在图外预计算并注入为 _cached_cos/_cached_sin。
    图内不再产生 Cast(206us) + MatMul + Cat + Cos + Sin 共 5 个 kernel。
    """
    return self._cached_cos, self._cached_sin


# ==================== 统一入口 ====================

def apply_fusion_ops():
    """应用全部融合算子替换 (monkey-patch)。

    在模型加载后、torch.compile 前调用。
    patch 作用于 transformers.models.qwen2.modeling_qwen2 模块级类/函数,
    因此对已加载的模型立即生效。
    """
    # 1. RMSNorm
    modeling_qwen2.Qwen2RMSNorm.forward = _npu_rms_norm_forward
    print("[fusion] Qwen2RMSNorm.forward → npu_rms_norm")

    # 2. RoPE rotate
    modeling_qwen2.apply_rotary_pos_emb = _npu_apply_rotary_pos_emb
    print("[fusion] apply_rotary_pos_emb → npu_apply_rotary_pos_emb")

    # 3. RoPE cos/sin 图外预计算
    modeling_qwen2.Qwen2RotaryEmbedding.forward = _npu_rotary_emb_forward
    print("[fusion] Qwen2RotaryEmbedding.forward → 图外预计算 cos/sin")


