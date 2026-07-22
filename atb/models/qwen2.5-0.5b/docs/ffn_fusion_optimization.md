# FFN 融合算子优化分析报告

> 测试日期: 2026-07-22
> 测试环境: Ascend910_9382, NPU device 0, 单线程
> 模型: qwen2.5-0.5b (varlen, 2D TND, frozen_parameter, fp16)
> 固定输入: batch=10, seq_len=208, total_tokens=2080

## 1. 问题发现

### 1.1 Profiling 初步观察

单线程 profiling 中, `aclnnFFNV3_FFN_FFN` (npu_ffn V3) 是耗时最大的算子, 占总执行时间 54%。

| 指标 | 值 |
|------|------|
| 算子 | `aclnnFFNV3_FFN_FFN` (npu_ffn, activation=swiglu) |
| 输入 | x=[2080,896], weight1=[896,9728], weight2=[4864,896] |
| 输出 | [2080,896], FP16 |
| 单次执行 | 389 us (wait time≈1us, 无排队) |
| 单线程耗时占比 | **54%** |

### 1.2 资源利用率异常

| 指标 | 值 | 含义 |
|------|------|------|
| aic_mac_ratio | **43.5%** | Cube MAC (矩阵乘) 实际计算占比 |
| aic_mte2_ratio | 72.7% | HBM→L1 数据加载占比 (主瓶颈) |
| aic_scalar_ratio | 76.3% | 标量单元 (地址计算/tiling逻辑) |
| aiv_vec_ratio | 9.3% | Vector Core (SwiGLU 激活) 利用率极低 |
| cube_utilization | 95.5% | Cube 整体忙碌率 (含数据搬运) |

Cube 95.5% 的时间都在忙, 但其中只有 43.5% 在做 MAC, 剩余 56.5% 在搬数据 (MTE2=72.7%)。
**FFN 是访存瓶颈, 不是计算瓶颈。**

### 1.3 npu_ffn API 文档确认

查阅 [npu_ffn 官方文档](https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/torchnpuCustomsapi/docs/zh/custom_APIs/torch_npu/torch_npu-npu_ffn.md):

- 文档为 API 参考手册, **不做性能推荐**, 只定义功能约束
- 当前用法 (FP16, swiglu, inner_precise=1, 无专家) 完全符合文档约束
- swiglu 激活仅支持无专家的 FP16 高性能模式, 无其他可选配置
- 43.5% 的 MAC 利用率是硬件执行特性, 非文档讨论范畴

## 2. 优化方案: 拆分 FFN 为小算子

### 2.1 原始 FFN 融合逻辑

代码位于 `qwen_varlen/fusion_ops.py`, `npu_ffn` 将整个 MLP 融合为 1 个算子:

```
npu_ffn(x, w1, w2, activation='swiglu') → 1 个算子
  内部: MatMul1(x@w1)→[2080,9728] → SwiGLU→[2080,4864] → MatMul2(@w2)→[2080,896]
  中间结果留在片上不回写 HBM
```

### 2.2 拆分后的小算子

不融合时, `Qwen2MLP.forward` 保持 transformers 原始实现:

```python
gate = self.gate_proj(x)     # MatMul  [2080,896]@[896,4864] → [2080,4864]
up   = self.up_proj(x)       # MatMul  [2080,896]@[896,4864] → [2080,4864]
act  = silu(gate) * up       # Swish+Mul (Vector), GE 融合为 EltwiseBroadcastFusionOp
out  = self.down_proj(inter) # MatMul  [2080,4864]@[4864,896] → [2080,896]
```

每层 3 MatMul + 1 EltwiseBroadcastFusionOp, 24 层共 72 MatMul + 24 Vector。

### 2.3 代码改动

#### `qwen_varlen/fusion_ops.py`
- `apply_fusion_ops()` 增加 `fuse_ffn=True` 参数
- `fuse_ffn=False` 时不 patch `Qwen2MLP.forward`, 保留原始小算子实现

#### `qwen_varlen/export_air.py`
- 增加 `--no-ffn-fusion` CLI flag
- `export_air()` 透传 `fuse_ffn` 给 `apply_fusion_ops()`
- `verify_air()` 增加 `fuse_ffn` 分支, 调整期望算子数

#### `atb/bench_latency.cpp`
- 增加 `--fixed-seq <len>` 选项, 固定所有序列长度 (用于公平 profiling 对比)

#### `atb/tools/parse_profiling.py`
- 修复重复导出问题: parse/export 阶段检查已导出结果, 存在则 skip
- 增加 `--force` 标志允许强制重新处理

### 2.4 导出验证

```
=== AIR 算子验证 (no-ffn-fusion) ===
  FusedInferAttentionScore              24  [OK]
  FFN                                    0  [OK]  (不应出现)
  RmsNorm                               49  [OK]
  ApplyRotaryPosEmb                     24  [OK]
  MatMulV2                              72  [OK]  (Q/K/V projection)
  MatMul                                97  [WARN] (O_proj+lm_head+gate+up+down = 24+1+24+24+24)
  Mul                                   24  [OK]  (SwiGLU: silu(gate)*up)
  Swish                                 24  (SiLU 被映射为 Swish, GE 进一步融合 Swish+Mul 为 EltwiseBroadcastFusionOp)
```

## 3. 测试结果

### 3.1 Benchmark 对比 (固定 batch=10, seq=208, 单线程, 70 请求)

| 指标 | 融合 FFN | 不融合 FFN | 提升 |
|------|----------|------------|------|
| **Execute avg** | 16.706 ms | **15.487 ms** | **-7.3%** |
| E2E avg | 17.626 ms | 16.142 ms | -8.4% |
| QPS | 56.68 | 61.92 | +9.3% |
| Token Throughput | 117,887 | 128,786 | +9.3% |

### 3.2 Profiling 对比 (per-iteration, 24 层 MLP)

| MLP 部分 | 融合 FFN | 不融合 FFN | 提升 |
|----------|----------|------------|------|
| gate/up MatMul (×48) | — | 4,881 us (101.7us/op) | — |
| Swish+Mul (×24) | — | 884 us (36.8us/op) | — |
| down MatMul (×24) | — | 1,989 us (82.9us/op) | — |
| **MLP 合计** | **9,276 us** | **7,754 us** | **-16.4%** |

### 3.3 MAC 利用率对比

| 算子 | MAC ratio | MTE2 ratio | CubeUtil | 说明 |
|------|-----------|------------|----------|------|
| **融合 FFN** | **43.5%** | 72.7% | 95.5% | tiling 妥协, MAC 空转多 |
| 拆分 gate/up MatMul | **85.4%** | 91.8% | 85.4% | 独立 tiling, MAC 翻倍 |
| 拆分 down MatMul | **83.4%** | 95.4% | 83.3% | 同上 |

### 3.4 全模型对比

| 指标 | 融合 FFN | 不融合 FFN | 提升 |
|------|----------|------------|------|
| 全算子总耗时/iter | 17,188 us | 15,302 us | **-11.0%** |

## 4. 根因分析

### 4.1 为什么拆分反而更快

初始分析预测拆分会慢 40%, 实际反而快 16.4%。预测错误的根本原因:

| 预测假设 | 实际情况 |
|----------|----------|
| 融合 FFN 的 tiling 与独立 MatMul 相当 | 融合 FFN tiling 妥协, MAC 仅 43.5% |
| 中间量 HBM 往返 (76us) 是主要开销 | tiling 优化的收益远超 HBM 往返代价 |
| 独立 MatMul MAC ratio ~50% | 实际达 85%, 远超预期 |

### 4.2 融合 FFN tiling 劣势

`npu_ffn` 内部需要处理两个不同 shape 的 MatMul:
- MatMul1: [2080,896] @ [896,9728] → [2080,9728] (K=896, N=9728)
- SwiGLU: [2080,9728] → [2080,4864]
- MatMul2: [2080,4864] @ [4864,896] → [2080,896] (K=4864, N=896)

两个 MatMul 的 K 和 N 差异大 (9728 vs 896), 融合 kernel 需要找到一种 tiling 同时适配两者, 导致妥协。拆分后每个 MatMul 独立 tiling, 可以针对各自 shape 优化:
- gate/up: K=896, N=4864 → MAC 85.4%
- down: K=4864, N=896 → MAC 83.4%

### 4.3 Swish+Mul 的额外开销可忽略

GE 编译器将 SiLU + Mul 自动融合为 `EltwiseBroadcastFusionOp` (1 个 Vector kernel, 36.8us/op), 总计 884us/iter, 仅占 MLP 总耗时的 11.4%。

## 5. 结论

1. **不融合 FFN 在此场景下更快 16.4%** (MLP 部分), 全模型提升 11.0%
2. **根因是 npu_ffn V3 的 tiling 策略对该 shape 不优**, MAC 利用率仅 43.5%, 拆分后独立 MatMul 达 85%
3. **中间量 HBM 往返的代价远小于 tiling 优化的收益**, 初始分析高估了访存开销
4. **Swish+Mul 被 GE 自动融合**, 额外开销仅 884us/iter, 可忽略

## 6. Profiling 数据位置

| 版本 | PROF 目录 |
|------|-----------|
| 融合 FFN (单线程) | `PROF_000001_20260720185221194_01838292EIHLKEKB` |
| 不融合 FFN (单线程, 固定 seq=208) | `PROF_000001_20260722192607889_00922330ERPFGDDO` |
| 不融合 FFN 打包 | `/export/home/weinan5/hejun/tran/profiling_no_ffn_single_thread.tar.gz` |

## 7. 复现步骤

```bash
# 1. 导出 AIR + ATC 编译 (不融合 FFN)
python -m qwen_varlen.export_air \
    --dynamic --run-atc --verify \
    --no-ffn-fusion \
    --model-name qwen2.5-0.5b-no-ffn \
    --device 0

# 2. 单线程 benchmark + profiling (固定 seq=208)
./build/bench_latency \
    --model models/qwen2.5-0.5b/om/qwen2.5-0.5b-no-ffn_linux_aarch64.om \
    --threads 1 --requests 70 --warmup 10 \
    --fixed-seq 208 --profiling --device-id 0

# 3. 解析 profiling (已支持 skip 已导出数据)
python3 tools/parse_profiling.py parse-and-export \
    --profiling_dir models/qwen2.5-0.5b/profiling_data

# 4. 融合 FFN 版本对比 (复测)
./build/bench_latency \
    --model models/qwen2.5-0.5b/om/qwen2.5-0.5b_linux_aarch64.om \
    --threads 1 --requests 70 --warmup 10 \
    --fixed-seq 208 --device-id 0
```
