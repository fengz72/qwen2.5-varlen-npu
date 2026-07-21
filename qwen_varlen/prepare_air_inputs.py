"""
为 AIR 导出的 OM 模型生成 4 个用户输入数据 + golden logits。

frozen_parameter=1 后, FFN 权重和 atten_mask 已冻结为图常量,
OM 只有 4 个图输入 (Data 节点):
  - actual_seq_lengths [N] int64
  - cos [1, T, 64] float16
  - sin [1, T, 64] float16
  - input_ids [T] int64

(position_ids 虽是 forward 参数, 但因 cos/sin 图外预计算而在图中消除)

cos/sin 预计算策略:
  1. 图外预计算 max_seq_len 的 cos/sin 表 [1, MAX_SEQ_LEN, 64] (一次)
  2. 运行时按 position_ids gather: cos_table[:, pos, :] → [1, T, 64]
     varlen 中每条 seq 的 position 从 0 重启: pos = [0..207, 0..207, ...]
  3. 图输入 shape 不变 [1, -1, 64], OM 模型无需重新导出

同时运行 eager 模式生成 golden logits 供精度对比。
"""

import os
import torch
import torch_npu
from transformers import AutoModelForCausalLM

from qwen_varlen.attention import register_npu_fia
from qwen_varlen.varlen_utils import generate_varlen_inputs, setup_varlen_attention, precompute_rope_cos_sin
from qwen_varlen.fusion_ops import apply_fusion_ops
from qwen_varlen.export_air import patch_attention_for_dynamic

DEFAULT_MODEL_PATH = "/export/home/models/Qwen2.5-0.5B"
DEFAULT_OUTPUT_DIR = "atb/models/qwen2.5-0.5b/input_data"
DEFAULT_DEVICE = 2

BATCH_SIZE = 10
SEQ_LEN = 208
MAX_SEQ_LEN = 2048


def main():
    torch.npu.set_device(DEFAULT_DEVICE)

    # 1. 加载模型 (与 export 完全一致)
    register_npu_fia()
    model = AutoModelForCausalLM.from_pretrained(
        DEFAULT_MODEL_PATH, dtype=torch.float16, attn_implementation="npu_fia"
    ).npu()
    model.eval()
    model.config.use_cache = False
    apply_fusion_ops(model)
    patch_attention_for_dynamic()

    # 2. 准备 varlen 输入 (全 0 token, 不需要 tokenizer)
    concat_ids, concat_pos, seq_lens, cum_seq_lens = generate_varlen_inputs(
        BATCH_SIZE, SEQ_LEN
    )
    setup_varlen_attention(model, cum_seq_lens, 'npu')

    print(f"seq_lens: {seq_lens[:5]}, cum_seq_lens: {cum_seq_lens}")

    # 2.1 预计算 max_seq_len 的 cos/sin 表, 按 position_ids gather (varlen 每条 seq 从 0 重启)
    precompute_rope_cos_sin(model, MAX_SEQ_LEN, 'npu')
    cos_table = model.model.rotary_emb._cached_cos  # [1, MAX_SEQ_LEN, 64]
    sin_table = model.model.rotary_emb._cached_sin
    pos = concat_pos.squeeze(0).npu()  # [T] = [0..207, 0..207, ...]
    model.model.rotary_emb._cached_cos = cos_table[:, pos, :]  # [1, T, 64]
    model.model.rotary_emb._cached_sin = sin_table[:, pos, :]
    print(f"cos/sin gathered: table=[1,{MAX_SEQ_LEN},64] → pos={pos.shape} → cos={model.model.rotary_emb._cached_cos.shape}")

    # 3. 生成 golden logits (eager 模式, 与导出路径完全一致: 仅最后 token + norm 后置)
    print("=== 生成 golden logits (仅每条序列最后一个 token) ===")
    asl_tensor = torch.tensor(cum_seq_lens, dtype=torch.int64, device='npu')
    with torch.no_grad():
        m = model.model
        hidden = m.embed_tokens(concat_ids.squeeze(0).npu())
        position_embeddings = m.rotary_emb(hidden, concat_pos.squeeze(0).npu())
        for layer in m.layers:
            hidden = layer(
                hidden,
                attention_mask=None,
                position_embeddings=position_embeddings,
                position_ids=concat_pos.squeeze(0).npu(),
                use_cache=False,
            )
        last_indices = asl_tensor - 1
        last_hidden = hidden.index_select(0, last_indices)
        last_hidden = m.norm(last_hidden)
        golden_logits = model.lm_head(last_hidden).cpu()
    print(f"golden_logits shape: {golden_logits.shape}")

    # 4. 收集 4 个用户输入 (按 OM Data 节点顺序)
    inputs = [
        ("actual_seq_lengths", torch.tensor(cum_seq_lens, dtype=torch.int64).cpu()),
        ("cos", model.model.rotary_emb._cached_cos.cpu()),
        ("sin", model.model.rotary_emb._cached_sin.cpu()),
        ("input_ids", concat_ids.squeeze(0).cpu()),
    ]

    # 5. 保存
    os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)

    list_lines = []
    arg_names = ["arg1_1", "arg3_1", "arg5_1", "arg8_1"]
    for idx, (name, tensor) in enumerate(inputs):
        fname = f"{name}.bin"
        fpath = os.path.join(DEFAULT_OUTPUT_DIR, fname)
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

    list_path = os.path.join(DEFAULT_OUTPUT_DIR, "input_list.txt")
    with open(list_path, 'w') as f:
        for line in list_lines:
            f.write(line + "\n")
    print(f"\nInput list saved to: {list_path} ({len(inputs)} inputs)")

    # 6. 保存 golden logits
    golden_path = os.path.join(DEFAULT_OUTPUT_DIR, "golden_logits.bin")
    golden_logits.detach().numpy().tofile(golden_path)
    print(f"Golden logits saved to: {golden_path}")


if __name__ == "__main__":
    main()
