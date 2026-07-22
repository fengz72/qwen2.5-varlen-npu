# Qwen2.5-0.5B 延迟与吞吐测试报告 (独立线程闭环模型)

> 测试日期: 2026-07-21
> 测试环境: Ascend910_9382, NPU device 14
> 模型: qwen2.5-0.5b (varlen, 2D TND, frozen_parameter, fp16)
> 测试工具: `atb/build/bench_latency`

## 1. 测试背景

### 1.1 Bug 修复说明

本次测试修复了 `bench_latency.cpp` 中 `RequestGenerator` 的一个 bug: `seq_log_dist_` 成员未用 `SEQ_LOG_MEAN` / `SEQ_LOG_STD` 初始化, 导致其使用 `std::normal_distribution` 的默认参数 (mean=0, stddev=1), 实际生成的 seq_len 服从 lognormal(0, 1) 而非预期的 lognormal(4.997, 0.167)。

| | 修复前 (bug) | 修复后 |
|---|---|---|
| 分布 | lognormal(0, 1) | lognormal(4.997, 0.167) |
| 平均 seq_len | ~1.87 | ~150 |
| 每请求 total_tokens (batch=10) | ~19 | ~1500 |

修复前所有测试数据均不可靠 (Execute 仅 ~5ms, Token/s 仅 ~3700), 修复后与 `main.cpp` profiling 的 ~17ms (batch=10, seq_len=208, total_tokens=2080) 吻合。

### 1.2 与 bench_throughput 的区别

| 特性 | bench_throughput (开环) | bench_latency (闭环) |
|------|------------------------|---------------------|
| 模型 | Producer-Consumer | N 个独立线程, 各自循环 |
| 数据生成 | Producer 统一生成入队 | 每个线程自己生成 |
| 队列 | 有界阻塞队列 | 无队列 |
| D2H | 无 | 有 (完整 E2E) |
| 延迟含义 | 含 Queue Wait | 无 Queue Wait, 纯单请求耗时 |
| 吞吐含义 | 压满场景的最大 QPS | N 个独立请求源的实际吞吐 |

### 1.3 测试目标

- 测量单请求真实 E2E 延迟 (含 D2H, 无队列等待)
- 拆分延迟为 Data Gen / H2D / Execute / D2H 四阶段
- 找到独立线程模型下的吞吐饱和点与延迟拐点
- 与 bench_throughput 结果对比, 评估队列等待的影响

### 1.4 架构

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

### 1.5 延迟拆分

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
device_id:         14
线程数 sweep:      1, 2, 3, 4, 5, 6, 7, 8
每线程独立模型实例: 是 (避免动态 shape 并发冲突)
```

## 3. 测试结果

### 3.1 完整 Sweep 数据 (1-8 threads)

| Threads | QPS | Token/s | Gen avg | H2D avg | Exec avg | Exec p99 | D2H avg | E2E avg | E2E p99 |
|---------|-------|---------|---------|---------|----------|----------|---------|---------|---------|
| 1 | 67.30 | 100,903 | 0.263 | 0.081 | 14.40 | 15.65 | 0.108 | 14.86 | 16.17 |
| 2 | 79.31 | 118,854 | 0.267 | 0.089 | 24.73 | 26.57 | 0.124 | 25.21 | 27.09 |
| 3 | 79.73 | 119,498 | 0.274 | 0.099 | 37.10 | 40.35 | 0.131 | 37.60 | 40.88 |
| **4** | **80.01** | **119,874** | 0.279 | 0.104 | 49.44 | 53.96 | 0.140 | 49.96 | 54.52 |
| 5 | 80.00 | 119,853 | 0.279 | 0.104 | 61.94 | 68.27 | 0.140 | 62.47 | 68.79 |
| 6 | 79.46 | 119,138 | 0.280 | 0.105 | 74.94 | 82.74 | 0.138 | 75.46 | 83.33 |
| 7 | 79.18 | 118,706 | 0.281 | 0.106 | 87.81 | 96.01 | 0.139 | 88.33 | 96.58 |
| 8 | 79.07 | 118,546 | 0.281 | 0.108 | 100.58 | 107.88 | 0.140 | 101.11 | 108.40 |

### 3.2 关键指标趋势

#### QPS 增长率 (相对 1 thread)

| Threads | QPS | 相对 1 thread | 边际增益 |
|---------|-----|--------------|---------|
| 1 | 67.30 | 1.00× | - |
| 2 | 79.31 | 1.18× | +12.01 |
| 3 | 79.73 | 1.19× | +0.42 |
| 4 | 80.01 | 1.19× | +0.28 |
| 5 | 80.00 | 1.19× | -0.01 |
| 6 | 79.46 | 1.18× | -0.54 |
| 7 | 79.18 | 1.18× | -0.28 |
| 8 | 79.07 | 1.17× | -0.11 |

#### Execute 时间线性增长

| Threads | Exec avg | 相对 1 thread |
|---------|----------|--------------|
| 1 | 14.40ms | 1.00× |
| 2 | 24.73ms | 1.72× |
| 3 | 37.10ms | 2.58× |
| 4 | 49.44ms | 3.43× |
| 5 | 61.94ms | 4.30× |
| 6 | 74.94ms | 5.20× |
| 7 | 87.81ms | 6.10× |
| 8 | 100.58ms | 6.98× |

#### 各阶段占比 (以 4 threads 为例)

| 阶段 | avg (ms) | 占 E2E 比例 |
|------|---------|------------|
| Data Gen | 0.279 | 0.56% |
| H2D | 0.104 | 0.21% |
| **Execute** | **49.44** | **98.95%** |
| D2H | 0.140 | 0.28% |
| **E2E** | **49.96** | 100% |

## 4. 分析

### 4.1 吞吐饱和点: 2 threads

```
1 thread:  67 QPS
2 threads: 79 QPS  (+18%)
3 threads: 80 QPS  (+0.5%)
4 threads: 80 QPS  (+0.4%)
5 threads: 80 QPS  (+0%)
6 threads: 79 QPS  (-0.6%)
7 threads: 79 QPS  (-0.3%)
8 threads: 79 QPS  (-0.1%)
```

NPU 计算资源在 **2 threads 时已接近打满**。2→8 threads QPS 仅从 79.31 增至 80.01 (+0.9%), 边际收益可忽略。4 threads 达到 QPS 峰值 80.01, 之后开始下降。

### 4.2 延迟线性劣化

E2E avg 随线程数近乎线性增长:

```
1 thread:  14.86ms   (独占 NPU)
4 threads: 49.96ms   (×3.4)
8 threads: 101.11ms  (×6.8)
```

说明 2 threads 后设备已满载, 额外并发只是排队等待, 不产生并行收益。

### 4.3 Execute 主导 E2E

Execute 占 E2E 的 **96%+**, 是绝对瓶颈:

| Threads | E2E avg | Execute avg | Execute 占比 |
|---------|---------|-------------|-------------|
| 1 | 14.86ms | 14.40ms | 96.9% |
| 4 | 49.96ms | 49.44ms | 99.0% |
| 8 | 101.11ms | 100.58ms | 99.5% |

Data Gen / H2D / D2H 合计占比 <4%, 且随线程数增长几乎不变, 说明这些阶段的并行性较好。

### 4.4 独立线程 vs Producer-Consumer 对比

> **注意**: `bench_throughput` 存在同样的 seq_len bug, 以下旧数据不可靠, 待修复后重新测试对比。

| Threads/Streams | 闭环 QPS (bench_latency, 修复后) | 开环 QPS (bench_throughput, 旧数据待重测) | 闭环 E2E avg | 开环 E2E avg (旧) |
|-----------------|----------------------------------|------------------------------------------|-------------|-------------------|
| 1 | 67.30 | 待重测 | 14.86ms | 待重测 |
| 4 | 80.01 | 待重测 | 49.96ms | 待重测 |
| 8 | 79.07 | 待重测 | 101.11ms | 待重测 |

### 4.5 D2H 开销分析

D2H 将 `[N, vocab]` fp16 输出 (batch=10, 151936 vocab) 从 device 拷回 host:

```
D2H 数据量 = 10 × 151936 × 2 = 3.04 MB
D2H avg   = 0.108-0.140ms (随线程数微增)
```

D2H 占 E2E 比例 <1%, 对整体延迟影响可忽略。

### 4.6 各阶段随线程数的变化

| Threads | Gen avg | H2D avg | Exec avg | D2H avg |
|---------|---------|---------|----------|---------|
| 1 | 0.263 | 0.081 | 14.40 | 0.108 |
| 4 | 0.279 | 0.104 | 49.44 | 0.140 |
| 8 | 0.281 | 0.108 | 100.58 | 0.140 |

- **Data Gen**: 几乎不变 (0.263→0.281ms), CPU 侧无竞争
- **H2D**: 微增 (0.081→0.108ms), PCIe 带宽有轻微竞争
- **Execute**: 线性增长 (14.40→100.58ms), NPU 算力分时复用
- **D2H**: 微增 (0.108→0.140ms), PCIe 带宽有轻微竞争

### 4.7 Data Gen 开销分析

修复后 Data Gen 从 ~0.005ms 增至 ~0.263ms, 因为 cos/sin gather 的数据量从 ~19 tokens 增至 ~1500 tokens (×80):

```
Data Gen = 采样 + gather cos/sin [total_tokens, 64]
         = ~1500 × 64 × 2 (cos+sin) × float→fp16 转换
         ≈ 0.263ms (1 thread)
```

Data Gen 占 E2E 比例 <2%, 对整体延迟影响可忽略。

## 5. 配置建议

### 5.1 独立线程模型最优配置: 2 threads

| Threads | QPS | E2E avg | E2E p99 | 评价 |
|---------|-----|---------|---------|------|
| 1 | 67 | 15ms | 16ms | 延迟最低, 吞吐未打满 |
| **2** | **79** | **25ms** | **27ms** | **吞吐接近峰值, 延迟可控** |
| 4 | 80 | 50ms | 55ms | QPS 峰值, 但延迟翻倍 |
| 8 | 79 | 101ms | 108ms | 延迟劣化严重, 无吞吐增益 |

选择 2 threads 的理由:
1. **QPS 达峰值的 99%**: 79.31 vs 80.01 (差 0.9%, 可忽略)
2. **E2E p99 仅 27ms**: 相比 4 threads 的 55ms 低 51%
3. **Exec p99 26.6ms**: 单请求执行快, 响应迅速
4. **超过 2 threads 收益递减**: QPS +0.9%, E2E p99 +104%

### 5.2 与 bench_throughput 建议对比

> `bench_throughput` 存在同样的 seq_len bug, 待修复后重新测试。旧数据不可靠, 此处暂不对比。

### 5.3 场景建议

| 场景 | 推荐 threads | 理由 |
|------|-------------|------|
| **单请求低延迟** (在线交互) | 1 | E2E p99 16ms, QPS 67 |
| **吞吐+延迟平衡** (通用服务) | 2 | QPS 79, E2E p99 27ms |
| **吞吐优先** (离线批量) | 4 | QPS 80 (峰值), E2E p99 55ms |
| **资源受限** | 1-2 | 避免 HBM 不足导致 OOM |

### 5.4 一句话总结

独立线程模型下, **2 threads 是 NPU 资源利用率与单请求延迟的最优平衡点**, QPS 达峰值 99%, E2E p99 仅 27ms。Execute 占 E2E 97%+, 是唯一值得优化的阶段。

## 6. 测试命令

### 6.1 单次测试

```bash
atb/build/bench_latency \
    --model atb/models/qwen2.5-0.5b/om/qwen2.5-0.5b_linux_aarch64.om \
    --threads 2 --requests 8000 --warmup 50 --device-id 14
```

### 6.2 Sweep 多个 thread 数

```bash
atb/build/bench_latency \
    --model atb/models/qwen2.5-0.5b/om/qwen2.5-0.5b_linux_aarch64.om \
    --sweep 1,2,3,4,5,6,7,8 --requests 8000 --warmup 50 --device-id 14
```

## 7. 附录 A: 各线程数详细数据

### 7.1 1 thread

```
Requests:         8000 (warmup=50)
Total Time:       118869.93 ms
Total Tokens:     11994331
QPS:              67.30 req/s
Token Throughput: 100903 tokens/s

Latency Breakdown (ms):
  E2E (full)   avg=14.857  p50=14.667  p90=15.731  p99=16.170  max=16.995
    Data Gen   avg=0.263   p50=0.261   p90=0.282   p99=0.316   max=0.781
    H2D        avg=0.081   p50=0.079   p90=0.089   p99=0.148   max=0.537
    Execute    avg=14.404  p50=14.216  p90=15.259  p99=15.653  max=16.485
    D2H        avg=0.108   p50=0.105   p90=0.117   p99=0.150   max=0.780
```

### 7.2 2 threads

```
Requests:         8000 (warmup=50)
Total Time:       100873.67 ms
Total Tokens:     11989283
QPS:              79.31 req/s
Token Throughput: 118854 tokens/s

Latency Breakdown (ms):
  E2E (full)   avg=25.213  p50=25.108  p90=26.337  p99=27.094  max=28.140
    Data Gen   avg=0.267   p50=0.264   p90=0.290   p99=0.319   max=0.495
    H2D        avg=0.089   p50=0.081   p90=0.118   p99=0.188   max=0.554
    Execute    avg=24.732  p50=24.621  p90=25.834  p99=26.568  max=27.647
    D2H        avg=0.124   p50=0.119   p90=0.145   p99=0.174   max=0.766

Per-Thread E2E:
  Thread 0: 4000 reqs, avg_e2e=25.212ms
  Thread 1: 4000 reqs, avg_e2e=25.215ms
```

### 7.3 3 threads

```
Requests:         8001 (warmup=50)
Total Time:       100353.03 ms
Total Tokens:     11991964
QPS:              79.73 req/s
Token Throughput: 119498 tokens/s

Latency Breakdown (ms):
  E2E (full)   avg=37.603  p50=37.232  p90=39.754  p99=40.877  max=64.286
    Data Gen   avg=0.274   p50=0.273   p90=0.298   p99=0.335   max=0.581
    H2D        avg=0.099   p50=0.093   p90=0.122   p99=0.202   max=0.446
    Execute    avg=37.097  p50=36.730  p90=39.225  p99=40.349  max=63.785
    D2H        avg=0.131   p50=0.130   p90=0.153   p99=0.189   max=0.874

Per-Thread E2E:
  Thread 0: 2667 reqs, avg_e2e=37.598ms
  Thread 1: 2667 reqs, avg_e2e=37.589ms
  Thread 2: 2667 reqs, avg_e2e=37.623ms
```

### 7.4 4 threads

```
Requests:         8000 (warmup=50)
Total Time:       99987.52 ms
Total Tokens:     11985894
QPS:              80.01 req/s
Token Throughput: 119874 tokens/s

Latency Breakdown (ms):
  E2E (full)   avg=49.964  p50=49.473  p90=52.704  p99=54.523  max=57.346
    Data Gen   avg=0.279   p50=0.277   p90=0.301   p99=0.349   max=0.512
    H2D        avg=0.104   p50=0.102   p90=0.122   p99=0.214   max=0.484
    Execute    avg=49.440  p50=48.946  p90=52.165  p99=53.961  max=56.881
    D2H        avg=0.140   p50=0.138   p90=0.164   p99=0.199   max=0.797

Per-Thread E2E:
  Thread 0: 2000 reqs, avg_e2e=49.986ms
  Thread 1: 2000 reqs, avg_e2e=49.933ms
  Thread 2: 2000 reqs, avg_e2e=49.952ms
  Thread 3: 2000 reqs, avg_e2e=49.987ms
```

### 7.5 5 threads

```
Requests:         8000 (warmup=50)
Total Time:       100004.95 ms
Total Tokens:     11985878
QPS:              80.00 req/s
Token Throughput: 119853 tokens/s

Latency Breakdown (ms):
  E2E (full)   avg=62.468  p50=61.881  p90=66.085  p99=68.789  max=72.267
    Data Gen   avg=0.279   p50=0.276   p90=0.301   p99=0.379   max=0.723
    H2D        avg=0.104   p50=0.101   p90=0.123   p99=0.222   max=0.549
    Execute    avg=61.944  p50=61.359  p90=65.526  p99=68.265  max=71.713
    D2H        avg=0.140   p50=0.137   p90=0.166   p99=0.199   max=4.131

Per-Thread E2E:
  Thread 0: 1600 reqs, avg_e2e=62.495ms
  Thread 1: 1600 reqs, avg_e2e=62.456ms
  Thread 2: 1600 reqs, avg_e2e=62.462ms
  Thread 3: 1600 reqs, avg_e2e=62.482ms
  Thread 4: 1600 reqs, avg_e2e=62.446ms
```

### 7.6 6 threads

```
Requests:         8004 (warmup=50)
Total Time:       100724.64 ms
Total Tokens:     12000098
QPS:              79.46 req/s
Token Throughput: 119138 tokens/s

Latency Breakdown (ms):
  E2E (full)   avg=75.464  p50=74.995  p90=79.356  p99=83.325  max=89.245
    Data Gen   avg=0.280   p50=0.277   p90=0.303   p99=0.371   max=0.860
    H2D        avg=0.105   p50=0.101   p90=0.125   p99=0.203   max=0.480
    Execute    avg=74.938  p50=74.471  p90=78.787  p99=82.739  max=88.623
    D2H        avg=0.138   p50=0.135   p90=0.166   p99=0.192   max=1.891

Per-Thread E2E:
  Thread 0: 1334 reqs, avg_e2e=75.486ms
  Thread 1: 1334 reqs, avg_e2e=75.357ms
  Thread 2: 1334 reqs, avg_e2e=75.495ms
  Thread 3: 1334 reqs, avg_e2e=75.497ms
  Thread 4: 1334 reqs, avg_e2e=75.487ms
  Thread 5: 1334 reqs, avg_e2e=75.460ms
```

### 7.7 7 threads

```
Requests:         8001 (warmup=50)
Total Time:       101048.14 ms
Total Tokens:     11995045
QPS:              79.18 req/s
Token Throughput: 118706 tokens/s

Latency Breakdown (ms):
  E2E (full)   avg=88.334  p50=88.046  p90=91.674  p99=96.582  max=103.911
    Data Gen   avg=0.281   p50=0.278   p90=0.306   p99=0.368   max=0.751
    H2D        avg=0.106   p50=0.101   p90=0.127   p99=0.202   max=0.579
    Execute    avg=87.806  p50=87.518  p90=91.116  p99=96.009  max=103.415
    D2H        avg=0.139   p50=0.135   p90=0.167   p99=0.202   max=0.894

Per-Thread E2E:
  Thread 0: 1143 reqs, avg_e2e=88.366ms
  Thread 1: 1143 reqs, avg_e2e=88.333ms
  Thread 2: 1143 reqs, avg_e2e=88.325ms
  Thread 3: 1143 reqs, avg_e2e=88.393ms
  Thread 4: 1143 reqs, avg_e2e=88.381ms
  Thread 5: 1143 reqs, avg_e2e=88.329ms
  Thread 6: 1143 reqs, avg_e2e=88.208ms
```

### 7.8 8 threads

```
Requests:         8000 (warmup=50)
Total Time:       101174.04 ms
Total Tokens:     11993785
QPS:              79.07 req/s
Token Throughput: 118546 tokens/s

Latency Breakdown (ms):
  E2E (full)   avg=101.115  p50=100.904  p90=103.904  p99=108.398  max=117.111
    Data Gen   avg=0.281    p50=0.278    p90=0.305    p99=0.377    max=0.813
    H2D        avg=0.108    p50=0.102    p90=0.130    p99=0.220    max=0.508
    Execute    avg=100.584  p50=100.373  p90=103.352  p99=107.877  max=116.605
    D2H        avg=0.140    p50=0.135    p90=0.170    p99=0.209    max=0.889

Per-Thread E2E:
  Thread 0: 1000 reqs, avg_e2e=101.131ms
  Thread 1: 1000 reqs, avg_e2e=101.079ms
  Thread 2: 1000 reqs, avg_e2e=101.020ms
  Thread 3: 1000 reqs, avg_e2e=101.154ms
  Thread 4: 1000 reqs, avg_e2e=101.158ms
  Thread 5: 1000 reqs, avg_e2e=101.117ms
  Thread 6: 1000 reqs, avg_e2e=101.103ms
  Thread 7: 1000 reqs, avg_e2e=101.156ms
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
- Device: NPU 14
- HBM: 65536 MB

### 8.3 测试工具

- 二进制: `atb/build/bench_latency`
- 源码: `atb/bench_latency.cpp`
- 模型: 独立线程 (无队列), N 个独立线程各自循环
- 每线程独立模型实例 (避免动态 shape 并发冲突)
- 延迟拆分: Data Gen / H2D / Execute / D2H
