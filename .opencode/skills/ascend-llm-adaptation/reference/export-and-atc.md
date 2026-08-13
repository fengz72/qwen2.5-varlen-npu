# 导出与 ATC 编译

## dynamo_export（PyTorch → AIR）

```python
from torch_npu.dynamo.torchair import dynamo_export, CompilerConfig

config = CompilerConfig()
config.experimental_config.frozen_parameter = 1  # 冻结权重为常量

dynamo_export(
    input_ids, position_ids, actual_seq_lengths, cos, sin,
    model=export_model,
    export_path=output_dir,
    export_name=model_name,
    dynamic=True,
    config=config,
)
```

## ATC 编译（AIR → OM）

```bash
atc --framework=1 \
    --model=model.air \
    --output=model \
    --soc_version=<soc_version>
```

- `--framework=1`：AIR 格式（GE 原生图格式）
- `--soc_version`：由 Step 0 的 `torch_npu.npu.get_device_name()` 获取
- 动态 shape 已编码在 AIR 中（由 `mark_dynamic` 标记）
- 默认 `force_fp16` 精度模式
- 输出：`model_linux_aarch64.om`（含全部冻结权重，大小取决于模型参数量）

## 生成测试输入 + golden logits

用 eager 模型生成 `.bin` 输入文件和 golden logits，用于精度验证：

```python
import torch
import torch_npu

# 1. 构造测试输入（varlen 场景：拼接 token + 计算 ASL）
input_ids, position_ids, actual_seq_lengths, cos, sin = build_test_inputs(...)
# 将每个输入 tensor 保存为 .bin
for name, tensor in [("input_ids", input_ids), ("actual_seq_lengths", actual_seq_lengths),
                     ("cos", cos), ("sin", sin)]:
    tensor.cpu().numpy().tofile(f"{name}.bin")

# 2. eager forward 生成 golden logits
with torch.no_grad():
    golden = model(input_ids, position_ids, actual_seq_lengths, cos, sin)
golden.cpu().numpy().tofile("golden_logits.bin")
```

## ACL 推理验证

用 CANN 自带 `ais_bench` 工具加载 OM 推理（shape 值需替换为实际模型参数）：

```bash
# ais_bench 随 CANN toolkit 安装，支持单文件/批量推理
python -m ais_bench.infer --model model.om \
    --input actual_seq_lengths.bin cos.bin sin.bin input_ids.bin \
    --input_shapes "<N>:int64:ND;1,<T>,<D>:float16:ND;1,<T>,<D>:float16:ND;<T>:int64:ND" \
    --output_dir ./output \
    --device 0
```

也可用 ACL C++ / Python API 自行编写推理脚本。对比 `output_0.bin` 和
`golden_logits.bin` 验证精度。

## baseline benchmark

用 `ais_bench` 做 benchmark（`--loop` 控制请求数，`--warmup` 预热）：

```bash
python -m ais_bench.infer --model model.om \
    --input actual_seq_lengths.bin cos.bin sin.bin input_ids.bin \
    --input_shapes "<N>:int64:ND;1,<T>,<D>:float16:ND;1,<T>,<D>:float16:ND;<T>:int64:ND" \
    --output_dir ./output \
    --device 0 \
    --loop 100 \
    --warmup 10
```

记录 baseline 指标：QPS、Execute avg、E2E avg、token 吞吐。

## 已有 AIR 时直接编译 OM

如果已有 AIR 文件，跳过 dynamo_export，直接执行 ATC：

```bash
atc --framework=1 \
    --model=<path/to/model>.air \
    --output=<path/to/output> \
    --soc_version=<soc_version>
```
