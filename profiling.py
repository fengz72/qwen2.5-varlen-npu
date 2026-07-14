"""
Profiling 工具 — warmup + 正式采集的封装
"""

import torch
import torch_npu
from torch_npu.profiler import profile, ProfilerActivity, tensorboard_trace_handler


def run_with_profiling(compiled_model, inputs, output_dir="./prof_output"):
    """先 warmup 触发图编译, 再正式采集 profiling 数据。

    Args:
        compiled_model: torch.compile 编译后的模型
        inputs: 模型输入 dict
        output_dir: profiling 输出目录

    Returns:
        logits: 正式推理的输出 logits
    """
    # ---- Warmup：触发图编译，不采集 profiling ----
    print("=== Warmup (触发图编译，不采集) ===")
    with torch.no_grad():
        _ = compiled_model(**inputs)
    torch.npu.synchronize()
    print("=== Warmup 完成 ===\n")

    # ---- 正式采集 ----
    print("=== Profiling 采集中 ===")
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.NPU],
        on_trace_ready=tensorboard_trace_handler(output_dir),
    ):
        with torch.no_grad():
            logits = compiled_model(**inputs).logits
        torch.npu.synchronize()
    print(f"=== Profiling 采集完成, 数据保存到 {output_dir} ===\n")

    return logits
