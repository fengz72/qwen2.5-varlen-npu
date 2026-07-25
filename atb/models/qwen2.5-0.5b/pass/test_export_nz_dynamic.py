"""
test.py 的图模式版本 (动态): torchair -> AIR -> OM -> profiling
M 维度动态, 验证 npu_format_cast 的 FRACTAL_NZ 格式能否在动态导出后保留。
"""

import os
import subprocess
import glob
import re
import csv
from collections import Counter

import numpy
import torch
import torch.nn as nn
import torch_npu
from torch_npu.dynamo.torchair import dynamo_export, CompilerConfig

torch.npu.set_device(8)
torch.npu.config.allow_internal_format = True

M = 16
K = 896
N = 1024


class MatMulModel(nn.Module):
    def __init__(self):
        super().__init__()
        weight = torch.rand((K, N), dtype=torch.float16).npu()
        self.register_buffer('weight', weight)

    def forward(self, x):
        return torch.matmul(x, self.weight)

model = MatMulModel().eval()

with torch.no_grad():
    model.weight = torch_npu.npu_format_cast(model.weight, torch_npu.Format.FRACTAL_NZ)
fmt = torch_npu.get_npu_format(model.weight)
print(f"weight format after npu_format_cast: {fmt}")
print(f"weight type: {type(model.weight)}")

x = torch.rand((M, K), dtype=torch.float16).npu()

# 动态: M 维度标记为动态
torch._dynamo.mark_dynamic(x, 0)

config = CompilerConfig()
config.experimental_config.frozen_parameter = 1

output_dir = "/tmp/opencode/test_nz_dynamic"
os.makedirs(output_dir, exist_ok=True)

print(f"=== dynamo_export (dynamic, M dim dynamic) ===")
dynamo_export(
    x,
    model=model,
    export_path=output_dir,
    export_name="test_nz_dynamic",
    dynamic=True,
    config=config,
)

air_path = os.path.join(output_dir, "test_nz_dynamic.air")
om_output = os.path.join(output_dir, "test_nz_dynamic")

# ATC with dynamic input_shape
input_shape = "arg1_1:-1,896"
cmd = (
    f"atc --framework=1"
    f" --model={air_path}"
    f" --output={om_output}"
    f" --soc_version=Ascend910_9382"
    f' --input_shape="{input_shape}"'
)

env = os.environ.copy()
numpy_site = os.path.dirname(os.path.dirname(numpy.__file__))
env['PYTHONPATH'] = f"{numpy_site}:{env.get('PYTHONPATH', '')}"

print(f"=== ATC: {cmd} ===")
result = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env)
print(result.stdout[-500:] if result.stdout else "")
if result.returncode != 0:
    print(f"[ERROR] ATC failed:\n{result.stderr[-1000:]}")
    exit(1)

om_file = om_output + "_linux_aarch64.om"
if not os.path.exists(om_file):
    candidates = glob.glob(f"{om_output}*.om")
    om_file = candidates[0] if candidates else None

if not om_file:
    print("[ERROR] OM not generated")
    exit(1)

print(f"=== OM: {om_file} ({os.path.getsize(om_file)/1024/1024:.1f} MB) ===")

print("\n=== Verify AIR ops ===")
pbtxt_path = os.path.join(output_dir, "dynamo.pbtxt")
if os.path.exists(pbtxt_path):
    with open(pbtxt_path) as f:
        content = f.read()
    op_counts = Counter(re.findall(r'op: "([^"]+)"', content))
    print(f"  AIR op counts: {dict(op_counts)}")
