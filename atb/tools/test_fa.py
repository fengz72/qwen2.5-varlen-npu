import torch
import torch_npu
from functools import partial
import numpy as np
import torchair

datatype=torch.float16
#my_rand_fp32 = partial(torch.randn, dtype=torch.float32)
#my_rand_uint8 = partial(torch.randint, dtype=torch.uint8)
def generate_input_data(file_path):
    data = np.load(file_path)
    tensor = torch.from_numpy(data).to(datatype)
    return tensor


tensor_li = [
[generate_input_data('./dump_data_npy_static/FlashAttentionScore.FusedFlashAttention_PartitionedCall__encoder_layer_0_attention_self_Transpose_3_Transpose_163.14.46.1781687406450784.input.0.npy'), 
 generate_input_data('./dump_data_npy_static/FlashAttentionScore.FusedFlashAttention_PartitionedCall__encoder_layer_0_attention_self_Transpose_3_Transpose_163.14.46.1781687406450784.input.1.npy'),
  generate_input_data('./dump_data_npy_static/FlashAttentionScore.FusedFlashAttention_PartitionedCall__encoder_layer_0_attention_self_Transpose_3_Transpose_163.14.46.1781687406450784.input.2.npy'),
   torch.from_numpy(np.load('./dump_data_npy_static/FlashAttentionScore.FusedFlashAttention_PartitionedCall__encoder_layer_0_attention_self_Transpose_3_Transpose_163.14.46.1781687406450784.input.3.npy')), 0, 0.125, 12],

]

stream1 = torch.npu.current_stream()
experimental_config = torch_npu.profiler._ExperimentalConfig(profiler_level=torch_npu.profiler.ProfilerLevel.Level2)

class NpuFlashAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, atten_mask, sparse_mode, scale, head_num):
        print('aaaa')
        return torch_npu.npu_fusion_attention(q, k,v, head_num=head_num, input_layout='BSH', scale=scale, sparse_mode=sparse_mode, atten_mask=atten_mask)

config = torchair.CompilerConfig()
npu_backend = torchair.get_npu_backend(compiler_config=config)

model = NpuFlashAttention()
model = torch.compile(model, backend=npu_backend, dynamic=False)

B = 80
S = 114
N = 12
D = 64
for q,k,v,atten_mask,sparse_mode,scale,head_num in tensor_li:
    q = q.reshape(B,S,N* D)
    v = v.reshape(B,S,N* D)
    k = k.reshape(B,S,N* D)
    # k = k.reshape(30,54,256)
    # v = v.reshape(30,54,192)
    # v = torch.nn.functional.pad(v.reshape(30,54,4,48), (0,16), mode='constant', value=0)
    # v = v.reshape(30,54,256)
    # split_pses = torch.split(pse.reshape(120,1,1,54), 30, dim=0)
    
    # print(pse)
    out = model(q, k, v, atten_mask, sparse_mode, scale, head_num)
      #  split_tensors = torch.split(out[0], 30, dim=0)
      #  result = torch.cat(split_tensors, dim=2)
    fa_out = out[0]
    # fa_out = torch.narrow(out[0].reshape(B,S,N,D), dim=3, start=0, length=48).reshape(30,1,192)
    result = fa_out
    print('fa output shape: ')
    print(result.shape)
    print('fa output: ')
    print(result)
