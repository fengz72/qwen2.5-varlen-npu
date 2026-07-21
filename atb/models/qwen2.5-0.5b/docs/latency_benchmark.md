# Qwen2.5-0.5B 延迟与吞吐测试报告 (独立线程闭环模型)

> 测试日期: 2026-07-21
> 测试环境: Ascend910_9382, NPU device 2
> 模型: qwen2.5-0.5b (varlen, 2D TND, frozen_parameter, fp16)
> 测试工具: `atb/build/bench_latency`

## 1. 测试背景

### 1.1 与 bench_throughput 的区别

| 特性 | bench_throughput (开环) | bench_latency (闭环) |
|------|------------------------|---------------------|
| 模型 | Producer-Consumer | N 个独立线程, 各自循环 |
| 数据生成 | Producer 统一生成入队 | 每个线程自己生成 |
| 队列 | 有界阻塞队列 | 无队列 |
| D2H | 无 | 有 (完整 E2E) |
| 延迟含义 | 含 Queue Wait | 无 Queue Wait, 纯单请求耗时 |
| 吞吐含义 | 压满场景的最大 QPS | N 个独立请求源的实际吞吐 |

### 1.2 测试目标

- 测量单请求真实 E2E 延迟 (含 D2H, 无队列等待)
- 拆分延迟为 Data Gen / H2D / Execute / D2H 四阶段
- 找到独立线程模型下的吞吐饱和点与延迟拐点
- 与 bench_throughput 结果对比, 评估队列等待的影响

### 1.3 架构

```
main()
  ├── 初始化 ACL + Context + cos/sin 表
  ├── for each thread count in sweep:
  │     ├── 创建 N 个 StreamContext (各自加载模型)
  │     ├── 启动 N 个线程, 每个线程独立执行:
  │     │     for (i = 0; i < requests_per_thread; i++) {
  │     │       arrive = now()
  │     │       generate(req)          // 数据生成
  │     │       H2D                    // 拷贝到 device
  │     │       set_dynamic_shape
  │     │       execute (async + sync) // NPU 推理
  │     │       D2H                    // 拷回 host
  │     │       record(latency)        // 记录
  │     │     }
  │     └── join 所有线程, 汇总报告
  └── 打印 sweep 对比表
```

### 1.4 延迟拆分

| 时间点 | 阶段 | 说明 |
|--------|------|------|
| arrive → gen_done | Data Gen | 采样 + gather cos/sin |
| gen_done → h2d_done | H2D | memcpy host→device + set_dynamic_shape |
| h2d_done → execute_done | Execute | aclmdlExecuteAsync + sync |
| execute_done → d2h_done | D2H | memcpy device→host |
| arrive → d2h_done | E2E | 完整端到端 |

## 2. 测试配置

```
模型:              qwen2.5-0.5b_linux_aarch64.om
batch_size:        固定 10
seq_len:           lognormal(4.997, 0.167), clipped [1, 218]  → avg≈150, p99≈218
cos/sin:           预计算 [218, 64] 表, 按 position_ids gather
warmup:            50 requests
total_requests:    8000 (每个线程 requests/threads 次)
device_id:         2
线程数 sweep:      1, 2, 3, 4, 5, 6, 7, 8
每线程独立模型实例: 是 (避免动态 shape 并发冲突)
```

## 3. 测试结果

### 3.1 完整 Sweep 数据 (1-8 threads)

| Threads | QPS | Token/s | Gen avg | H2D avg | Exec avg | Exec p99 | D2H avg | E2E avg | E2E p99 |
|---------|-------|---------|---------|---------|----------|----------|---------|---------|---------|
| 1 | 196.48 | 3,683 | 0.005 | 0.050 | 4.92 | 5.78 | 0.112 | 5.09 | 5.96 |
| 2 | 239.64 | 4,493 | 0.005 | 0.055 | 8.17 | 9.05 | 0.111 | 8.34 | 9.24 |
| 3 | 254.37 | 4,760 | 0.006 | 0.059 | 11.59 | 12.61 | 0.122 | 11.78 | 12.82 |
| 4 | 256.20 | 4,798 | 0.007 | 0.072 | 15.38 | 16.74 | 0.133 | 15.60 | 16.96 |
| 5 | 258.23 | 4,840 | 0.007 | 0.074 | 19.12 | 20.81 | 0.138 | 19.34 | 21.06 |
| 6 | 258.50 | 4,873 | 0.007 | 0.076 | 22.97 | 25.03 | 0.138 | 23.19 | 25.25 |
| 7 | **260.63** | **4,908** | 0.007 | 0.077 | 26.61 | 28.66 | 0.142 | 26.84 | 28.93 |
| 8 | 260.04 | 4,896 | 0.008 | 0.077 | 30.52 | 32.55 | 0.142 | 30.75 | 32.77 |

### 3.2 关键指标趋势

#### QPS 增长率 (相对 1 thread)

| Threads | QPS | 相对 1 thread | 边际增益 |
|---------|-----|--------------|---------|
| 1 | 196.48 | 1.00× | - |
| 2 | 239.64 | 1.22× | +43.16 |
| 3 | 254.37 | 1.29× | +14.73 |
| 4 | 256.20 | 1.30× | +1.83 |
| 5 | 258.23 | 1.31× | +2.03 |
| 6 | 258.50 | 1.32× | +0.27 |
| 7 | 260.63 | 1.33× | +2.13 |
| 8 | 260.04 | 1.32× | -0.59 |

#### Execute 时间线性增长

| Threads | Exec avg | 相对 1 thread |
|---------|----------|--------------|
| 1 | 4.92ms | 1.00× |
| 2 | 8.17ms | 1.66× |
| 3 | 11.59ms | 2.36× |
| 4 | 15.38ms | 3.13× |
| 5 | 19.12ms | 3.89× |
| 6 | 22.97ms | 4.67× |
| 7 | 26.61ms | 5.41× |
| 8 | 30.52ms | 6.20× |

#### 各阶段占比 (以 4 threads 为例)

| 阶段 | avg (ms) | 占 E2E 比例 |
|------|---------|------------|
| Data Gen | 0.007 | 0.04% |
| H2D | 0.072 | 0.46% |
| **Execute** | **15.38** | **98.6%** |
| D2H | 0.133 | 0.85% |
| **E2E** | **15.60** | 100% |

## 4. 分析

### 4.1 吞吐饱和点: 4 threads

```
1 thread:  196 QPS
2 threads: 240 QPS  (+22%)
3 threads: 254 QPS  (+14%)
4 threads: 256 QPS  (+1%)
5 threads: 258 QPS  (+1%)
6 threads: 259 QPS  (+0%)
7 threads: 261 QPS  (+1%)  ← 峰值
8 threads: 260 QPS  (-0%)
```

NPU 计算资源在 **4 threads 时已接近打满**。4→8 threads QPS 仅从 256 增至 260 (+1.5%), 边际收益可忽略。7 threads 达到 QPS 峰值 261, 之后开始下降。

### 4.2 延迟线性劣化

E2E avg 随线程数近乎线性增长:

```
1 thread:  5.09ms   (独占 NPU)
4 threads: 15.60ms  (×3.1)
8 threads: 30.75ms  (×6.0)
```

说明 4 threads 后设备已满载, 额外并发只是排队等待, 不产生并行收益。

### 4.3 Execute 主导 E2E

Execute 占 E2E 的 **96%+**, 是绝对瓶颈:

| Threads | E2E avg | Execute avg | Execute 占比 |
|---------|---------|-------------|-------------|
| 1 | 5.09ms | 4.92ms | 96.7% |
| 4 | 15.60ms | 15.38ms | 98.6% |
| 8 | 30.75ms | 30.52ms | 99.2% |

Data Gen / H2D / D2H 合计占比 <4%, 且随线程数增长几乎不变, 说明这些阶段的并行性较好。

### 4.4 独立线程 vs Producer-Consumer 对比

| Threads/Streams | 闭环 QPS (bench_latency) | 开环 QPS (bench_throughput) | 闭环 E2E avg | 开环 E2E avg |
|-----------------|------------------------|---------------------------|-------------|-------------|
| 1 | 196.48 | 201.53 | 5.09ms | 29.71ms |
| 4 | 256.20 | 259.09 | 15.60ms | 80.61ms |
| 8 | 260.04 | 259.96 | 30.75ms | 156.21ms |

**关键发现:**

1. **QPS 接近**: 两种模型吞吐相当 (差异 <3%), 说明 NPU 算力是真正瓶颈, 任务调度方式影响不大。
2. **闭环 E2E 远低于开环**: 闭环无 Queue Wait, E2E = 纯单请求耗时; 开环含排队, E2E 被队列等待主导。
   - 1 thread: 闭环 5ms vs 开环 30ms (6×差距, 排队占 83%)
   - 8 threads: 闭环 31ms vs 开环 156ms (5×差距, 排队占 80%)
3. **闭环更能反映单请求真实延迟**, 开环更能反映压满场景下的用户感知延迟。

### 4.5 D2H 开销分析

D2H 将 `[N, vocab]` fp16 输出 (batch=10, 151936 vocab) 从 device 拷回 host:

```
D2H 数据量 = 10 × 151936 × 2 = 3.04 MB
D2H avg   = 0.11-0.14ms (随线程数微增)
```

D2H 占 E2E 比例 <1%, 对整体延迟影响可忽略。

### 4.6 各阶段随线程数的变化

| Threads | Gen avg | H2D avg | Exec avg | D2H avg |
|---------|---------|---------|----------|---------|
| 1 | 0.005 | 0.050 | 4.92 | 0.112 |
| 4 | 0.007 | 0.072 | 15.38 | 0.133 |
| 8 | 0.008 | 0.077 | 30.52 | 0.142 |

- **Data Gen**: 几乎不变 (0.005→0.008ms), CPU 侧无竞争
- **H2D**: 微增 (0.05→0.077ms), PCIe 带宽有轻微竞争
- **Execute**: 线性增长 (4.92→30.52ms), NPU 算力分时复用
- **D2H**: 微增 (0.11→0.14ms), PCIe 带宽有轻微竞争

## 5. 配置建议

### 5.1 独立线程模型最优配置: 4 threads

| Threads | QPS | E2E avg | E2E p99 | 评价 |
|---------|-----|---------|---------|------|
| 1 | 196 | 5ms | 6ms | 延迟最低, 吞吐未打满 |
| **4** | **256** | **16ms** | **17ms** | **吞吐接近峰值, 延迟可控** |
| 7 | 261 | 27ms | 29ms | QPS 峰值, 但延迟翻倍 |
| 8 | 260 | 31ms | 33ms | 延续劣化, 无吞吐增益 |

选择 4 threads 的理由:
1. **QPS 达峰值的 98%**: 256 vs 261 (差 2%, 可忽略)
2. **E2E p99 仅 17ms**: 相比 7 threads 的 29ms 低 41%
3. **Exec p99 16.7ms**: 单请求执行快, 响应迅速
4. **超过 4 threads 收益递减**: QPS +1.5%, E2E p99 +72%

### 5.2 与 bench_throughput 建议一致

两种测试模型均指向 **4 streams/threads 为最优配置**:

| 测试模型 | 推荐 | QPS | E2E p99 |
|---------|------|-----|---------|
| 闭环 (bench_latency) | 4 threads | 256 | 17ms |
| 开环 (bench_throughput) | 4 streams | 259 | 88ms |

闭环 E2E p99 (17ms) 远低于开环 (88ms), 因为闭环无队列等待。实际在线服务更接近开环模型 (请求需排队), 但闭环数据可用于评估单请求处理能力上限。

### 5.3 场景建议

| 场景 | 推荐 threads | 理由 |
|------|-------------|------|
| **单请求低延迟** (在线交互) | 1-2 | E2E p99 < 10ms |
| **吞吐+延迟平衡** (通用服务) | 4 | QPS 256, E2E p99 17ms |
| **吞吐优先** (离线批量) | 7-8 | QPS 峰值 261 |
| **资源受限** | 2-3 | QPS 254 (97% 峰值), 延迟可控 |

### 5.4 一句话总结

独立线程模型下, **4 threads 是 NPU 资源利用率与单请求延迟的最优平衡点**, QPS 达峰值 98%, E2E p99 仅 17ms。Execute 占 E2E 96%+, 是唯一值得优化的阶段。

## 6. 测试命令

### 6.1 单次测试

```bash
atb/build/bench_latency \
    --model atb/models/qwen2.5-0.5b/om/qwen2.5-0.5b_linux_aarch64.om \
    --threads 4 --requests 8000 --warmup 50 --device-id 2
```

### 6.2 Sweep 多个 thread 数

```bash
atb/build/bench_latency \
    --model atb/models/qwen2.5-0.5b/om/qwen2.5-0.5b_linux_aarch64.om \
    --sweep 1,2,3,4,5,6,7,8 --requests 8000 --warmup 50 --device-id 2
```

## 7. 附录 A: 各线程数详细数据

### 7.1 1 thread

```
Requests:         8000 (warmup=50)
Total Time:       40715.98 ms
Total Tokens:     149941
QPS:              196.48 req/s
Token Throughput: 3683 tokens/s

Latency Breakdown (ms):
  E2E (full)   avg=5.088  p50=5.054  p90=5.218  p99=5.965  max=6.897
    Data Gen   avg=0.005  p50=0.004  p90=0.007  p99=0.012  max=0.047
    H2D        avg=0.050  p50=0.050  p90=0.052  p99=0.062  max=1.107
    Execute    avg=4.920  p50=4.887  p90=5.047  p99=5.782  max=6.493
    D2H        avg=0.112  p50=0.110  p90=0.116  p99=0.128  max=0.826
```

### 7.2 2 threads

```
Requests:         8000 (warmup=50)
Total Time:       33383.39 ms
Total Tokens:     149984
QPS:              239.64 req/s
Token Throughput: 4493 tokens/s

Latency Breakdown (ms):
  E2E (full)   avg=8.343  p50=8.347  p90=8.572  p99=9.239  max=10.365
    Data Gen   avg=0.005  p50=0.004  p90=0.007  p99=0.013  max=0.051
    H2D        avg=0.055  p50=0.053  p90=0.058  p99=0.094  max=0.487
    Execute    avg=8.170  p50=8.178  p90=8.396  p99=9.046  max=9.939
    D2H        avg=0.111  p50=0.109  p90=0.120  p99=0.149  max=0.335

Per-Thread E2E:
  Thread 0: 4000 reqs, avg_e2e=8.341ms
  Thread 1: 4000 reqs, avg_e2e=8.344ms
```

### 7.3 3 threads

```
Requests:         8001 (warmup=50)
Total Time:       31454.66 ms
Total Tokens:     149733
QPS:              254.37 req/s
Token Throughput: 4760 tokens/s

Latency Breakdown (ms):
  E2E (full)   avg=11.776  p50=11.732  p90=12.223  p99=12.821  max=66.841
    Data Gen   avg=0.006   p50=0.005   p90=0.008   p99=0.015   max=0.192
    H2D        avg=0.059   p50=0.056   p90=0.066   p99=0.167   max=0.482
    Execute    avg=11.588  p50=11.543  p90=12.024  p99=12.613  max=66.611
    D2H        avg=0.122   p50=0.119   p90=0.138   p99=0.173   max=0.970

Per-Thread E2E:
  Thread 0: 2667 reqs, avg_e2e=11.753ms
  Thread 1: 2667 reqs, avg_e2e=11.784ms
  Thread 2: 2667 reqs, avg_e2e=11.792ms
```

### 7.4 4 threads

```
Requests:         8000 (warmup=50)
Total Time:       31225.27 ms
Total Tokens:     149825
QPS:              256.20 req/s
Token Throughput: 4798 tokens/s

Latency Breakdown (ms):
  E2E (full)   avg=15.597  p50=15.575  p90=16.208  p99=16.963  max=19.711
    Data Gen   avg=0.007   p50=0.006   p90=0.009   p99=0.019   max=0.221
    H2D        avg=0.072   p50=0.062   p90=0.082   p99=0.298   max=0.660
    Execute    avg=15.383  p50=15.367  p90=15.979  p99=16.739  max=19.279
    D2H        avg=0.133   p50=0.128   p90=0.160   p99=0.200   max=0.824

Per-Thread E2E:
  Thread 0: 2000 reqs, avg_e2e=15.593ms
  Thread 1: 2000 reqs, avg_e2e=15.581ms
  Thread 2: 2000 reqs, avg_e2e=15.604ms
  Thread 3: 2000 reqs, avg_e2e=15.610ms
```

### 7.5 5 threads

```
Requests:         8000 (warmup=50)
Total Time:       30980.03 ms
Total Tokens:     149956
QPS:              258.23 req/s
Token Throughput: 4840 tokens/s

Latency Breakdown (ms):
  E2E (full)   avg=19.345  p50=19.334  p90=20.152  p99=21.064  max=66.392
    Data Gen   avg=0.007   p50=0.006   p90=0.009   p99=0.068   max=0.224
    H2D        avg=0.074   p50=0.063   p90=0.084   p99=0.276   max=0.486
    Execute    avg=19.125  p50=19.116  p90=19.927  p99=20.813  max=66.145
    D2H        avg=0.138   p50=0.134   p90=0.167   p99=0.204   max=0.801

Per-Thread E2E:
  Thread 0: 1600 reqs, avg_e2e=19.345ms
  Thread 1: 1600 reqs, avg_e2e=19.340ms
  Thread 2: 1600 reqs, avg_e2e=19.326ms
  Thread 3: 1600 reqs, avg_e2e=19.355ms
  Thread 4: 1600 reqs, avg_e2e=19.359ms
```

### 7.6 6 threads

```
Requests:         8004 (warmup=50)
Total Time:       30963.17 ms
Total Tokens:     150882
QPS:              258.50 req/s
Token Throughput: 4873 tokens/s

Latency Breakdown (ms):
  E2E (full)   avg=23.191  p50=23.175  p90=24.156  p99=25.252  max=26.971
    Data Gen   avg=0.007   p50=0.006   p90=0.009   p99=0.059   max=0.184
    H2D        avg=0.076   p50=0.064   p90=0.091   p99=0.315   max=0.549
    Execute    avg=22.969  p50=22.955  p90=23.925  p99=25.029  max=26.779
    D2H        avg=0.138   p50=0.134   p90=0.166   p99=0.204   max=0.894

Per-Thread E2E:
  Thread 0: 1334 reqs, avg_e2e=23.193ms
  Thread 1: 1334 reqs, avg_e2e=23.191ms
  Thread 2: 1334 reqs, avg_e2e=23.161ms
  Thread 3: 1334 reqs, avg_e2e=23.207ms
  Thread 4: 1334 reqs, avg_e2e=23.194ms
  Thread 5: 1334 reqs, avg_e2e=23.202ms
```

### 7.7 7 threads

```
Requests:         8001 (warmup=50)
Total Time:       30699.11 ms
Total Tokens:     150673
QPS:              260.63 req/s
Token Throughput: 4908 tokens/s

Latency Breakdown (ms):
  E2E (full)   avg=26.840  p50=26.821  p90=27.864  p99=28.925  max=30.619
    Data Gen   avg=0.007   p50=0.006   p90=0.009   p99=0.068   max=0.190
    H2D        avg=0.077   p50=0.067   p90=0.092   p99=0.217   max=0.487
    Execute    avg=26.613  p50=26.595  p90=27.630  p99=28.659  max=30.269
    D2H        avg=0.142   p50=0.140   p90=0.167   p99=0.212   max=1.023

Per-Thread E2E:
  Thread 0: 1143 reqs, avg_e2e=26.851ms
  Thread 1: 1143 reqs, avg_e2e=26.826ms
  Thread 2: 1143 reqs, avg_e2e=26.853ms
  Thread 3: 1143 reqs, avg_e2e=26.840ms
  Thread 4: 1143 reqs, avg_e2e=26.820ms
  Thread 5: 1143 reqs, avg_e2e=26.854ms
  Thread 6: 1143 reqs, avg_e2e=26.839ms
```

### 7.8 8 threads

```
Requests:         8000 (warmup=50)
Total Time:       30763.97 ms
Total Tokens:     150606
QPS:              260.04 req/s
Token Throughput: 4896 tokens/s

Latency Breakdown (ms):
  E2E (full)   avg=30.749  p50=30.758  p90=31.718  p99=32.773  max=35.172
    Data Gen   avg=0.008   p50=0.006   p90=0.009   p99=0.058   max=0.556
    H2D        avg=0.077   p50=0.066   p90=0.093   p99=0.239   max=0.666
    Execute    avg=30.521  p50=30.532  p90=31.484  p99=32.552  max=34.961
    D2H        avg=0.142   p50=0.141   p90=0.167   p99=0.207   max=1.244

Per-Thread E2E:
  Thread 0: 1000 reqs, avg_e2e=30.734ms
  Thread 1: 1000 reqs, avg_e2e=30.747ms
  Thread 2: 1000 reqs, avg_e2e=30.759ms
  Thread 3: 1000 reqs, avg_e2e=30.755ms
  Thread 4: 1000 reqs, avg_e2e=30.758ms
  Thread 5: 1000 reqs, avg_e2e=30.751ms
  Thread 6: 1000 reqs, avg_e2e=30.742ms
  Thread 7: 1000 reqs, avg_e2e=30.749ms
```

## 8. 附录 B: 测试环境详情

### 8.1 模型信息

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

### 8.2 硬件环境

- SoC: Ascend910_9382
- CANN: 9.0.0
- Device: NPU 2
- HBM: 65536 MB

### 8.3 测试工具

- 二进制: `atb/build/bench_latency`
- 源码: `atb/bench_latency.cpp`
- 模型: Producer-Consumer (无队列), N 个独立线程各自循环
- 每线程独立模型实例 (避免动态 shape 并发冲突)
- 延迟拆分: Data Gen / H2D / Execute / D2H
