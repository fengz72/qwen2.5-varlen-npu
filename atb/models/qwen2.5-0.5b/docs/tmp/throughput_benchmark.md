# Qwen2.5-0.5B 多流吞吐性能测试报告

> 测试日期: 2026-07-21
> 测试环境: Ascend910_9382, NPU device 2
> 模型: qwen2.5-0.5b (varlen, 2D TND, frozen_parameter, fp16)

## 1. 测试背景

### 1.1 线上场景特征

| 指标 | 均值 | p99 |
|------|------|-----|
| batch_size | 9.8 | 10.9 |
| seq_len (tokens) | 150 | 218 |

### 1.2 测试目标

- 验证不同 stream 数下的吞吐 (QPS) 和延迟 (latency)
- 找到吞吐与延迟的最优平衡点
- 为瞬时高并发场景提供配置建议

### 1.3 测试工具

- `atb/build/bench_throughput` (C++ 实现, 无 GIL 限制)
- 架构: Producer-Consumer + 多 stream 并发
- 每个 stream 加载独立模型实例, 避免动态 shape 模型的并发冲突
- 延迟拆分: Prep (gather cos/sin) + Queue Wait + Inference

## 2. 测试配置

### 2.1 固定 batch_size 测试 (batch=10)

```
模型:          qwen2.5-0.5b_linux_aarch64.om
batch_size:    固定 10
seq_len:       lognormal(4.997, 0.167), clipped [1, 218]  → avg≈150, p99≈218
cos/sin:       预计算 [218, 64] 表, 按 position_ids gather (计入 Prep 时间)
warmup:        50 requests
total_requests: 2000
device_id:     2
queue_depth:   streams × 4
```

### 2.2 随机 batch_size 测试 (batch~normal(9.8,0.35))

测试配置同上, 仅 batch_size 改为正态分布模拟线上真实场景。结果见附录 A。

## 3. 测试结果 (固定 batch=10, 含 Prep 时间)

### 3.1 完整 Sweep 数据 (1-8 streams)

| Streams | QPS | Token/s | Prep avg | E2E avg | E2E p99 | Infer avg | Infer p99 | Queue Wait avg |
|---------|-------|---------|----------|---------|---------|-----------|-----------|----------------|
| 1 | 201.53 | 3,768 | 0.006 | 29.71 | 31.74 | 4.89 | 5.91 | 24.75 |
| 2 | 244.96 | 4,580 | 0.006 | 44.79 | 49.26 | 8.10 | 8.97 | 36.62 |
| 3 | 258.62 | 4,836 | 0.005 | 61.60 | 68.14 | 11.52 | 12.57 | 50.01 |
| 4 | 259.09 | 4,845 | 0.004 | 80.61 | 87.87 | 15.35 | 16.56 | 65.19 |
| 5 | **260.79** | **4,876** | 0.008 | 99.09 | 109.02 | 19.06 | 20.84 | 79.94 |
| 6 | 261.04 | 4,881 | 0.007 | 117.89 | 130.54 | 22.87 | 25.23 | 94.93 |
| 7 | 260.53 | 4,871 | 0.006 | 137.03 | 147.52 | 26.76 | 28.81 | 110.19 |
| 8 | 259.96 | 4,861 | 0.005 | 156.21 | 172.98 | 30.65 | 32.90 | 125.49 |

### 3.2 关键指标趋势

#### QPS 增长率 (相对 1 stream)

| Streams | QPS | 相对 1 stream | 边际增益 |
|---------|-----|--------------|---------|
| 1 | 201.53 | 1.00× | - |
| 2 | 244.96 | 1.22× | +43.43 |
| 3 | 258.62 | 1.28× | +13.66 |
| 4 | 259.09 | 1.29× | +0.47 |
| 5 | 260.79 | 1.29× | +1.70 |
| 6 | 261.04 | 1.30× | +0.25 |
| 7 | 260.53 | 1.29× | -0.51 |
| 8 | 259.96 | 1.29× | -0.57 |

#### Inference 时间线性增长

| Streams | Infer avg | 相对 1 stream |
|---------|-----------|--------------|
| 1 | 4.89ms | 1.00× |
| 2 | 8.10ms | 1.66× |
| 3 | 11.52ms | 2.36× |
| 4 | 15.35ms | 3.14× |
| 5 | 19.06ms | 3.90× |
| 6 | 22.87ms | 4.68× |
| 7 | 26.76ms | 5.47× |
| 8 | 30.65ms | 6.27× |

#### Prep 时间 (gather cos/sin)

| Streams | Prep avg (ms) | 占 E2E 比例 |
|---------|--------------|------------|
| 1 | 0.006 | 0.02% |
| 4 | 0.004 | 0.005% |
| 8 | 0.005 | 0.003% |

**Prep 时间可忽略** (<0.01ms), 相对 inference (5-30ms) 占比 <0.2%。

## 4. 分析

### 4.1 E2E 与 Inference 的关系

```
E2E latency = Prep + Queue Wait + Inference
```

以 6 streams 为例:
```
E2E avg      = 117.89 ms
Prep         =   0.01 ms   (gather cos/sin, 占 0.0%)
Queue Wait   =  94.93 ms   (等待空闲 stream, 占 80.5%)
Inference    =  22.87 ms   (实际 NPU 执行, 占 19.4%)
```

**多流场景下, E2E 延迟主要被 Queue Wait 主导**, Prep 和 Inference 合计仅占 20%。

### 4.2 QPS 在 3 streams 后趋于饱和

```
3 streams: 259 QPS
4 streams: 259 QPS  (+0.2%)
5 streams: 261 QPS  (+0.7%)
6 streams: 261 QPS  (+0.1%)  ← 峰值
7 streams: 261 QPS  (-0.2%)
8 streams: 260 QPS  (-0.2%)
```

NPU 计算资源在 **3 streams 时已接近打满**。6 streams 达到 QPS 峰值 261, 之后开始下降。更多 stream 主要增加排队延迟, 而非吞吐。

### 4.3 Inference 时间随 stream 线性增长

Infer time ≈ 4.89ms × streams (近似线性), 说明 NPU 资源在 stream 间**分时复用**:

```
1 stream:  4.89ms × 1 = 4.89ms   (独占 NPU)
4 streams: 4.89ms × 4 ≈ 19.6ms   (实际 15.4ms, 有部分重叠)
8 streams: 4.89ms × 8 ≈ 39.1ms   (实际 30.7ms, 有部分重叠)
```

实际增长略低于线性, 说明 stream 间有少量并行重叠 (如 H2D 传输与计算重叠)。

### 4.4 多流以延迟换吞吐

| 场景 | E2E p99 | QPS | 代价 |
|------|---------|-----|------|
| 1 stream | 32ms | 202 | 基线 |
| 4 streams | 88ms | 259 | E2E ×2.8, QPS ×1.29 |
| 6 streams | 131ms | 261 | E2E ×4.1, QPS ×1.30 |

**代价不对称**: E2E p99 增长 4 倍, QPS 仅增长 30%。

### 4.5 Prep 时间分析

```
Prep = gather cos/sin + 构建 input_ids + actual_seq_lengths
     ≈ 0.005-0.008 ms (5-8 微秒)
```

Prep 包含:
- 从 [218, 64] 预计算表中按 position_ids gather cos/sin
- fp32 → fp16 转换
- 构建 varlen 输入 (input_ids, actual_seq_lengths)

**结论: Prep 耗时可忽略**, batch=10/seq_len=150 下仅 6μs, 相对 inference (5-30ms) 占比 <0.2%。

### 4.6 固定 batch vs 随机 batch 对比

| Streams | QPS (固定10) | QPS (随机9.8) | 差异 |
|---------|-------------|--------------|------|
| 1 | 201.53 | 203.92 | -1.2% |
| 4 | 259.09 | 259.96 | -0.3% |
| 5 | 260.79 | 262.34 | -0.6% |
| 8 | 259.96 | 259.51 | +0.2% |

固定 batch=10 与随机 batch 的 QPS 差异 <1.2%, **两种配置的结论一致: 5-6 streams QPS 峰值, 4 streams 为延迟/吞吐平衡点**。

## 5. 配置建议

### 5.1 瞬时高并发场景: 推荐 4 streams

| Streams | 排空 2000 请求 | E2E p99 | Infer p99 |
|---------|---------------|---------|-----------|
| 3 | 7.73s | 68ms | 12.6ms |
| **4** | **7.72s** | **88ms** | **16.6ms** |
| 5 | 7.67s | 109ms | 20.8ms |
| 6 | 7.66s | 131ms | 25.2ms |

选择 4 streams 的理由:
1. **排空速度接近峰值**: 7.72s vs 最优 7.66s (差 0.06s, 可忽略)
2. **E2E p99 仅 88ms**: 相比 6 streams 的 131ms 低 33%
3. **Infer p99 16.6ms**: 单请求执行快, 积压消化快
4. **超过 4 streams 收益递减**: QPS 仅 +0.7% (6 streams), 但 E2E p99 +49%

瞬时高并发时 stream 越多, 单次 infer 越慢, 抵消并行收益。4 streams 是 NPU 资源利用率和单请求延迟的最优平衡点。

### 5.2 其他场景建议

| 场景 | 推荐 streams | 理由 |
|------|-------------|------|
| **吞吐优先** (离线批量) | 5-6 | QPS 峰值 261 |
| **延迟优先** (在线服务) | 2-3 | E2E p99 < 70ms |
| **瞬时高并发** | 4 | 排空快 + p99 可控 |
| **稳态低并发** | 1-2 | 避免资源浪费 |

### 5.3 一句话总结

吞吐在 3-4 streams 已打满, 再多就是用延迟换不到吞吐。**4 streams 兼顾排空速度和尾延迟, 适合瞬时高并发场景**。

## 6. 测试命令

### 6.1 单次测试

```bash
atb/build/bench_throughput \
    --model atb/models/qwen2.5-0.5b/om/qwen2.5-0.5b_linux_aarch64.om \
    --streams 4 --requests 2000 --warmup 50 --device-id 2
```

### 6.2 Sweep 多个 stream 数

```bash
atb/build/bench_throughput \
    --model atb/models/qwen2.5-0.5b/om/qwen2.5-0.5b_linux_aarch64.om \
    --sweep 1,2,3,4,5,6,7,8 --requests 2000 --warmup 50 --device-id 2
```

### 6.3 单流性能基线 (main.cpp)

```bash
./run.sh -m qwen2.5-0.5b --device-id 2 --skip-atc --warmup 10 --bench 100
```

## 7. 附录 A: 随机 batch_size 测试结果 (batch~normal(9.8,0.35))

| Streams | QPS | Token/s | E2E avg | E2E p99 | Infer avg | Infer p99 |
|---------|-------|---------|---------|---------|-----------|-----------|
| 1 | 203.92 | 3,774 | 29.30 | 31.08 | 4.84 | 5.93 |
| 2 | 245.29 | 4,540 | 44.61 | 48.95 | 8.09 | 8.88 |
| 3 | 258.04 | 4,751 | 61.72 | 67.67 | 11.55 | 12.55 |
| 4 | 259.96 | 4,787 | 80.40 | 88.98 | 15.32 | 16.49 |
| 5 | 262.34 | 4,831 | 98.52 | 107.14 | 18.98 | 20.79 |
| 6 | 262.63 | 4,836 | 117.18 | 126.54 | 22.76 | 24.64 |
| 7 | 261.74 | 4,820 | 136.32 | 147.93 | 26.63 | 28.96 |
| 8 | 259.51 | 4,803 | 155.10 | 177.14 | 30.70 | 32.96 |

> 随机 batch 与固定 batch=10 趋势一致, QPS 差异 <0.5%。

## 8. 附录 B: 算子级 Profiling (110 iterations, batch=10, seq_len=208)

| 算子 | Core 类型 | 次数 | 总耗时(us) | 平均(us) | 占比 |
|------|----------|------|-----------|---------|------|
| FFN | MIX_AIC | 2640 | 1,020,335.67 | 386.49 | 53.97% |
| FusedInferAttentionScore | MIX_AIC | 2640 | 329,872.79 | 124.95 | 17.45% |
| MatMulV2 | AI_CORE | 10670 | 259,866.76 | 24.36 | 13.74% |
| InplaceAddRmsNorm | AI_VECTOR_CORE | 5170 | 143,778.52 | 27.81 | 7.61% |
| ApplyRotaryPosEmb | AI_VECTOR_CORE | 2640 | 67,572.37 | 25.60 | 3.57% |
| TransData | AI_VECTOR_CORE | 330 | 53,561.21 | 162.31 | 2.83% |
| GatherV2 | AI_VECTOR_CORE | 220 | 10,830.86 | 49.23 | 0.57% |
| RmsNorm | AI_VECTOR_CORE | 220 | 3,427.27 | 15.58 | 0.18% |

单次迭代算子总耗时: ~16.77ms (batch=10, seq_len=208, total_tokens=2080)
单流推理实测: ~17.28ms (step_trace p50)

## 9. 附录 C: 测试环境详情

### 9.1 模型信息

- 模型: Qwen2.5-0.5B
- 参数量: 0.5B
- 层数: 24
- hidden_size: 896
- num_attention_heads: 14
- num_key_value_heads: 2
- head_dim: 64
- vocab_size: 151936
- 精度: fp16
- 导出: torchair.dynamo_export → AIR → ATC → OM
- 融合算子: FusedInferAttentionScore, FFN, RmsNorm, ApplyRotaryPosEmb

### 9.2 精度验证

OM 推理 vs eager golden logits:
- 余弦相似度: 0.99998529
- Pearson 相关系数: 0.99998592
- 最大绝对误差: 0.078
- 相对 L2 误差: 0.005

### 9.3 硬件环境

- SoC: Ascend910_9382
- CANN: 9.0.0
- Device: NPU 2
