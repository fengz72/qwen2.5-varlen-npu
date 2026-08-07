# GatherV2 vs SplitV+ConcatV2 单算子性能对比

> 测试日期: 2026-08-06
> 测试环境: Ascend910_9382, NPU device 8
> 测试脚本: `atb/tools/bench_gather_vs_split.py`
> 运行命令: `python -m atb.tools.bench_gather_vs_split --device 8`

## 1. 背景

Prefix caching 实现中, 每层 attention 前后各有一次 compact↔expanded 的数据搬运:

```
compact:  [prefix, req_0, req_1, ..., req_n]           (去重, 省 FFN 计算)
expanded: [prefix, req_0, prefix, req_1, ..., prefix, req_n]  (全 batch, 供 attention)
```

当前实现使用 `torch.index_select` (底层映射为 GatherV2), 在 profiling 中显示:
- 50 次/iter (24 层 × 2 + 首尾 2 次)
- ~14 us/call, 总计 ~670-750 us/iter, 占总 iter 时间 6.7%
- 净增 ~430 us/iter, 导致 prefix caching 为负优化

本文档测试一种替代方案 (SplitV+ConcatV2), 验证是否能降低搬运开销。

## 2. 测试配置

| 项目 | 值 |
|------|-----|
| D (hidden_size) | 896 |
| dtype | fp16 |
| avg_seq_len | 150 |
| p99_seq_len | 218 |
| prefix_len | 20-25 (随机) |
| N (batch_size) | 1, 3, 5, 8, 10 |
| warmup | 50 iters |
| measure | 1000 iters |
| seed | 42 |
| device | npu:8 |

### 2.1 变长序列生成

对齐线上真实分布 (`bench_throughput.py` 中的 lognormal 分布):

```python
sigma = 0.15
mu = np.log(150) - sigma**2 / 2
seqs = np.random.lognormal(mean=mu, sigma=sigma, size=N)
seqs = np.clip(seqs, 50, 218).astype(int)
```

### 2.2 两种方案

**方案 A — GatherV2 (当前实现)**:

```python
# expand: compact → expanded
expanded = torch.index_select(hidden, 0, expand_index)

# restore: expanded → compact
compact = torch.index_select(expanded, 0, restore_index)
```

- `expand_index` / `restore_index` 为 NPU tensor (int64), 运行时动态
- 底层映射为 GatherV2 算子, 1 次 kernel launch
- 逐 token 散列读: `output[i] = input[index[i]]`

**方案 B — SplitV+ConcatV2 (连续搬运方案)**:

```python
# expand: compact → expanded
prefix = hidden[:P]                                      # Slice
prefix_3d = prefix.unsqueeze(0).expand(N, P, D)          # BroadcastTo
reqs = hidden[P:]
req_splits = torch.split(reqs, [R_0, R_1, ...], dim=0)   # SplitV
expanded = torch.cat([prefix_3d[0], req_splits[0],
                      prefix_3d[1], req_splits[1], ...])  # ConcatV2

# restore: expanded → compact
block_splits = torch.split(expanded, [P+R_0, P+R_1, ...], dim=0)  # SplitV
prefix = block_splits[0][:P]                              # Slice
compact = torch.cat([prefix, block_splits[0][P:],
                     block_splits[1][P:], ...])            # ConcatV2
```

- N、P 为 Python int 常量 (编译期固定)
- R_i (各请求长度) 运行时动态, SplitV 的 size_splits 支持动态
- 每段内部连续, 走 DMA burst
- 算子节点数: expand = N+5 个/层, restore = N+3 个/层

## 3. 测试结果

### 3.1 单次算子耗时

| N | P | T_compact | T_expanded | expand_gather (us) | expand_split (us) | ratio | restore_gather (us) | restore_split (us) | ratio |
|---|---|-----------|------------|-------------------|-------------------|-------|---------------------|---------------------|-------|
| 1 | 25 | 218 | 218 | 8.65 | 39.34 | 4.55x | 8.66 | 30.21 | 3.49x |
| 3 | 25 | 472 | 522 | 10.16 | 45.57 | 4.49x | 10.02 | 37.50 | 3.74x |
| 5 | 25 | 751 | 851 | 9.85 | 53.24 | 5.41x | 9.86 | 43.06 | 4.37x |
| 8 | 25 | 1192 | 1367 | 9.57 | 58.51 | 6.11x | 9.56 | 49.13 | 5.14x |
| 10 | 25 | 1440 | 1665 | 10.40 | 65.41 | 6.29x | 10.88 | 56.62 | 5.20x |

### 3.2 24 层总耗时 (模拟真实推理)

每 iter 调用次数: 24 层 × (expand + restore) + 首尾 2 次 expand = 50 次

| N | gather 总计 (us) | split 总计 (us) | 差值 (us) | 倍数 |
|---|-----------------|-----------------|-----------|------|
| 1 | 432.8 | 1747.8 | +1315.0 | split 慢 4.0x |
| 3 | 504.7 | 2084.8 | +1580.1 | split 慢 4.1x |
| 5 | 492.8 | 2417.6 | +1924.8 | split 慢 4.9x |
| 8 | 478.3 | 2700.4 | +2222.1 | split 慢 5.6x |
| 10 | 531.5 | 3059.6 | +2528.1 | split 慢 5.8x |

### 3.3 正确性验证

所有配置下两种方案输出 `torch.allclose = True`, 验证 SplitV+ConcatV2 方案逻辑正确。

### 3.4 原始测试数据

```
[N=1]  P=25  T_c=218  T_e=218
  seq_lens=[218]
  expand:  gather=   8.65 us  split=  39.34 us  ratio=4.55x  correct=True
  restore: gather=   8.66 us  split=  30.21 us  ratio=3.49x  correct=True
  24-layer total (50 calls): gather=   432.8 us  split= 1747.8 us  diff= +1315.0 us

[N=3]  P=25  T_c=472  T_e=522
  seq_lens=[159, 145, 218]
  expand:  gather=  10.16 us  split=  45.57 us  ratio=4.49x  correct=True
  restore: gather=  10.02 us  split=  37.50 us  ratio=3.74x  correct=True
  24-layer total (50 calls): gather=   504.7 us  split= 2084.8 us  diff= +1580.1 us

[N=5]  P=25  T_c=751  T_e=851
  seq_lens=[159, 145, 218, 186, 143]
  expand:  gather=   9.85 us  split=  53.24 us  ratio=5.41x  correct=True
  restore: gather=   9.86 us  split=  43.06 us  ratio=4.37x  correct=True
  24-layer total (50 calls): gather=   492.8 us  split= 2417.6 us  diff= +1924.8 us

[N=8]  P=25  T_c=1192  T_e=1367
  seq_lens=[159, 145, 163, 186, 143, 218, 187, 166]
  expand:  gather=   9.57 us  split=  58.51 us  ratio=6.11x  correct=True
  restore: gather=   9.56 us  split=  49.13 us  ratio=5.14x  correct=True
  24-layer total (50 calls): gather=   478.3 us  split= 2700.4 us  diff= +2222.1 us

[N=10]  P=25  T_c=1440  T_e=1665
  seq_lens=[159, 145, 163, 186, 218, 143, 187, 166, 138, 160]
  expand:  gather=  10.40 us  split=  65.41 us  ratio=6.29x  correct=True
  restore: gather=  10.88 us  split=  56.62 us  ratio=5.20x  correct=True
  24-layer total (50 calls): gather=   531.5 us  split= 3059.6 us  diff= +2528.1 us
```

## 4. 分析

### 4.1 GatherV2 耗时基本不随 N 变化

| N | expand (us) | restore (us) |
|---|-------------|--------------|
| 1 | 8.65 | 8.66 |
| 3 | 10.16 | 10.02 |
| 5 | 9.85 | 9.86 |
| 8 | 9.57 | 9.56 |
| 10 | 10.40 | 10.88 |

GatherV2 始终只有 **1 次 kernel launch**, N 增大仅增加数据量, NPU vector core (48 核) 并行处理, 耗时稳定在 ~9-11 us。

### 4.2 SplitV+ConcatV2 耗时随 N 线性增长

| N | expand (us) | restore (us) | 算子节点数/层 |
|---|-------------|--------------|--------------|
| 1 | 39.34 | 30.21 | expand=6, restore=4 |
| 3 | 45.57 | 37.50 | expand=8, restore=6 |
| 5 | 53.24 | 43.06 | expand=10, restore=8 |
| 8 | 58.51 | 49.13 | expand=13, restore=11 |
| 10 | 65.41 | 56.62 | expand=15, restore=13 |

每层 Slice 数量 = `2N+2` (expand 中 N 个 prefix Slice + N 个 req SplitV output + 2 个主 Slice; restore 类似)。每个 Slice/SplitV/ConcatV2 都是独立的 kernel launch, launch 开销 ~1-2 us/次, 随 N 线性累积。

### 4.3 无 break-even 点

理论分析曾假设 N≥9 时 SplitV 可能反超 (因连续 DMA 优势)。实测证明不存在 break-even:

- GatherV2 1 次 launch 处理所有 token, vector core 并行
- SplitV 方案 2N+8 次 launch/层, launch 开销主导, 连续 DMA 收益无法补偿
- N 越大, SplitV 的算子节点越多, 差距反而越大 (N=1 慢 4x, N=10 慢 5.8x)

### 4.4 与 profiling 数据一致性

Profiling 实测 (N=10, seq=150, prefix=20): GatherV2 ~14 us/call, 50 次/iter = ~700 us/iter。
本次单算子测试 (N=10, 变长 seq avg=150): GatherV2 ~10 us/call, 50 次/iter = ~532 us。

差异原因: profiling 包含完整图执行时的内存竞争和流水线开销, 单算子测试环境更干净。两者量级一致, 互相印证。

## 5. 结论

1. **GatherV2 在所有 N 下都显著优于 SplitV+ConcatV2**, 差距 4-5.8 倍, 不存在 break-even 点
2. **SplitV+ConcatV2 的瓶颈是 kernel launch 次数** (2N+8 个/层), 而非数据搬运方式。连续 DMA 的收益远无法补偿 launch 开销
3. **GatherV2 的优势在于 1 次 launch 处理全部 token**, NPU vector core 并行度高, 散列读的开销被并行度掩盖
4. **正确性验证通过**, SplitV+ConcatV2 逻辑正确, 但性能不达标
5. **当前 GatherV2 方案应保留**, 优化方向应转向减少 GatherV2 调用次数 (如融合到相邻算子 RmsNorm+Gather) 或减少单次开销, 而非替换搬运方式

## 6. 附录: 测试脚本

脚本位置: `atb/tools/bench_gather_vs_split.py`

运行方式:
```bash
python -m atb.tools.bench_gather_vs_split --device 8
python -m atb.tools.bench_gather_vs_split --device 8 --n-list 1,3,5,8,10 --iters 2000
```
