"""
融合算子替换 — 通过 monkey-patch 将 Qwen2 小算子替换为 NPU 融合算子

替换项:
    1. RMSNorm:  Pow + ReduceMean + Add + Rsqrt + Mul + Cast → npu_rms_norm
    2. RoPE:     rotate_half(StridedSlice + Neg + Cat) + Mul + Add → npu_rotary_mul
    3. RoPE cos/sin: 图内 Cast(206us) + MatMul + Cos + Sin → 图外预计算注入
    4. FFN:      2 MatMul + Cat + SwiGLU + MatMul → npu_ffn (swiglu)

用法:
    from qwen_varlen.fusion_ops import apply_fusion_ops
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


# ==================== 2. RoPE rotate → npu_rotary_mul ====================

def _npu_apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """用 npu_rotary_mul 融合算子替代 rotate_half + Mul + Add。

    原始: cos.unsqueeze → rotate_half(StridedSlice+Neg+Cat) → Mul → Add (×2 for q,k)
    融合: npu_rotary_mul(q, cos, sin, 'half') → 1 个算子 (×2 for q,k)

    npu_rotary_mul 半旋转模式:
        x1, x2 = chunk(input, 2, -1)
        x_new = cat((-x2, x1), dim=-1)
        output = cos * input + sin * x_new
    等价于 transformers 的 rotate_half 实现。
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = torch_npu.npu_rotary_mul(q, cos, sin, 'half')
    k_embed = torch_npu.npu_rotary_mul(k, cos, sin, 'half')
    return q_embed, k_embed


# ==================== 3. RoPE cos/sin → 图外预计算 ====================

def _npu_rotary_emb_forward(self, x, position_ids):
    """直接返回图外预计算的 cos/sin, 避免图内 Cast/MatMul/Cos/Sin。

    cos/sin 由 setup_varlen_attention 在图外预计算并注入为 _cached_cos/_cached_sin。
    图内不再产生 Cast(206us) + MatMul + Cat + Cos + Sin 共 5 个 kernel。
    """
    return self._cached_cos, self._cached_sin


# ==================== 4. FFN → npu_ffn (swiglu) ====================

def _prepare_ffn_weights(model):
    """预拼接 MLP 权重, 注入到每个 layer。

    nn.Linear weight: [out_features, in_features]
    npu_ffn weight1: [K, N] = [hidden, 2*intermediate]
    npu_ffn weight2: [K, N] = [intermediate, hidden]

    swiglu 验证确认: weight1 = cat([up_w.T, gate_w.T], dim=1)
    npu_ffn 内部: silu(second_half) * first_half = silu(x@gate) * (x@up)
    """
    for layer in model.model.layers:
        mlp = layer.mlp
        w1 = torch.cat(
            [mlp.up_proj.weight.T, mlp.gate_proj.weight.T], dim=1
        ).contiguous()
        w2 = mlp.down_proj.weight.T.contiguous()
        mlp._ffn_w1 = w1
        mlp._ffn_w2 = w2


def _npu_ffn_forward(self, x):
    """用 npu_ffn 融合整个 MLP: 2 MatMul + Cat + SwiGLU + MatMul → 1 个算子。

    原始: gate_proj(x) → up_proj(x) → cat → swiglu → down_proj
    融合: npu_ffn(x, w1, w2, 'swiglu') → 1 个算子
    """
    return torch_npu.npu_ffn(
        x, self._ffn_w1, self._ffn_w2,
        activation='swiglu',
        inner_precise=1,
    )


# ==================== 统一入口 ====================

def apply_fusion_ops(model=None):
    """应用全部融合算子替换 (monkey-patch)。

    在模型加载后、torch.compile 前调用。

    Args:
        model: 可选, 传入模型实例用于校验。实际 patch 作用于
               transformers.models.qwen2.modeling_qwen2 模块级类/函数,
               因此对已加载的模型立即生效。
    """
    # 1. RMSNorm
    modeling_qwen2.Qwen2RMSNorm.forward = _npu_rms_norm_forward
    print("[fusion] Qwen2RMSNorm.forward → npu_rms_norm")

    # 2. RoPE rotate
    modeling_qwen2.apply_rotary_pos_emb = _npu_apply_rotary_pos_emb
    print("[fusion] apply_rotary_pos_emb → npu_rotary_mul")

    # 3. RoPE cos/sin 图外预计算
    modeling_qwen2.Qwen2RotaryEmbedding.forward = _npu_rotary_emb_forward
    print("[fusion] Qwen2RotaryEmbedding.forward → 图外预计算 cos/sin")

    # 4. FFN (npu_ffn 融合整个 MLP)
    if model is not None:
        _prepare_ffn_weights(model)
    modeling_qwen2.Qwen2MLP.forward = _npu_ffn_forward
    print("[fusion] Qwen2MLP.forward → npu_ffn (swiglu)")

    if model is not None:
        _patch_model_instances(model)


def _patch_model_instances(model):
    """对已实例化的模型, 替换其 forward 方法绑定。

    monkey-patch 类方法后, 已实例化的对象会自动引用新方法,
    但显式绑定可确保在极端情况下 (如方法被复制) 也生效。
    """
    import types

    # 遍历所有 decoder layer
    layers = getattr(model.model, 'layers', [])
    for layer in layers:
        # RMSNorm
        if hasattr(layer, 'input_layernorm'):
            layer.input_layernorm.forward = types.MethodType(
                _npu_rms_norm_forward, layer.input_layernorm
            )
        if hasattr(layer, 'post_attention_layernorm'):
            layer.post_attention_layernorm.forward = types.MethodType(
                _npu_rms_norm_forward, layer.post_attention_layernorm
            )
        # MLP
        if hasattr(layer, 'mlp'):
            layer.mlp.forward = types.MethodType(_npu_ffn_forward, layer.mlp)

    # 最终 norm
    if hasattr(model.model, 'norm'):
        model.model.norm.forward = types.MethodType(
            _npu_rms_norm_forward, model.model.norm
        )

    # Rotary embedding
    if hasattr(model.model, 'rotary_emb'):
        model.model.rotary_emb.forward = types.MethodType(
            _npu_rotary_emb_forward, model.model.rotary_emb
        )
