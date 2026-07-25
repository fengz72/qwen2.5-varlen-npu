import torch
import torch_npu

torch.npu.config.allow_internal_format = True
M = 16
K = 896
N = 151936
x1 = torch.rand((M, K)).to(torch.float16).npu()
x2 = torch.rand((K, N)).to(torch.float16).npu()
x2 = torch_npu.npu_format_cast(x2, 29)
out = torch.matmul(x1, x2)