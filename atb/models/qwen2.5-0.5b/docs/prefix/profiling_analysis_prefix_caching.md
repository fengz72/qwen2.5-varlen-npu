# Qwen2.5-0.5B Prefix Caching Profiling 分析

> 测试日期: 2026-08-05
> 测试环境: Ascend910_9382, NPU device 4, 24 AI cores, 48 vector cores, AIC/AIV 1800MHz
> 模型: qwen2.5-0.5b (24层, hidden=896, 14 q-heads, 2 kv-heads, head_dim=64, vocab=151936, fp16)
> 运行日志: `prefix.log`
> 对比基准: `profiling_analysis_seq150_vs_seq218.md` (无 prefix, device 8)

## 1. 测试配置

| 项目 | seq150+prefix20 | seq218+prefix25 |
|------|----------------|----------------|
| 命令 | `--fixed-seq 150 --prefix-len 20 --device 4` | `--fixed-seq 218 --prefix-len 25 --device 4` |
| Profiling 目录 | `PROF_...00830477NBEDAJHN` | `PROF_...00830700ADDCIDII` |
| batch_size | 10 | 10 |
| seq_len | 150 | 218 |
| prefix_len | 20 | 25 |
| 总 tokens/请求 | 1500 (=150×10) | 2180 (=218×10) |
| 唯一 tokens/请求 | 1320 (=20+130×10) | 1955 (=25+193×10) |
| 去重率 | 12.0% (180/1500) | 10.3% (225/2180) |
| warmup / requests | 50 / 100 | 50 / 100 |

### 1.1 Prefix Caching 机制

Prefix caching 将 batch 内共享的 prefix tokens 去重, 减少 embedding 查找和 FFN 计算量:

```
唯一 tokens = prefix_len + (seq_len - prefix_len) × batch
           = 20 + (150-20) × 10 = 1320  (seq150, prefix20)
           = 25 + (218-25) × 10 = 1955  (seq218, prefix25)
```

每层的计算模式:
```
1. Expand: 1320 (唯一) → 1500 (全batch)  [GatherV2]
2. Attention: 在 1500 tokens 上计算 Q/K/V/O + attention
3. Compress: 1500 (全batch) → 1320 (唯一)  [GatherV2]
4. FFN: 在 1320 tokens 上计算 Gate+Up+Down  ← 节省计算量
```

每层 2 次 GatherV2 (expand + compress), 加上初始和末尾各 1 次 expand, 共 50 次/iter:
- 1 次 vocab embedding gather (151936 → 唯一 tokens)
- 25 次 expand (唯一 → 全batch): 1 次 (layer 1 前) + 24 次 (每层 attention 前)
- 24 次 compress (全batch → 唯一): 24 次 (每层 attention 后, FFN 前)

## 2. 运行日志解析 (prefix.log)

### 2.1 seq150 + prefix20

```
Config: threads=1, total_requests=100, reqs/thread=100, warmup=50, device=4
Total Time:       1173.68 ms
Total Tokens:     150000
QPS:              85.20 req/s
Token Throughput: 127804 tokens/s

Latency Breakdown (ms):
  E2E (full)   avg=11.732   p50=11.707   p90=11.791   p99=12.594   max=12.594
    Data Gen   avg=0.274    p50=0.270    p90=0.290    p99=0.308    max=0.308
    H2D        avg=0.142    p50=0.137    p90=0.161    p99=0.244    max=0.244
    Execute    avg=11.200   p50=11.185   p90=11.230   p99=12.046   max=12.046
    D2H        avg=0.114    p50=0.113    p90=11.120   p99=0.175    max=0.175
```

### 2.2 seq218 + prefix25

```
Config: threads=1, total_requests=100, reqs/thread=100, warmup=50, device=4
Total Time:       1441.64 ms
Total Tokens:     218000
QPS:              69.37 req/s
Token Throughput: 151217 tokens/s

Latency Breakdown (ms):
  E2E (full)   avg=14.410   p50=14.354   p90=14.681   p99=15.096   max=15.096
    Data Gen   avg=0.394    p50=0.388    p90=0.413    p99=0.449    max=0.449
    H2D        avg=0.209    p50=0.189    p90=0.288    p99=0.501    max=0.501
    Execute    avg=13.683   p50=13.653   p90=13.776   p99=14.113   max=14.113
    D2H        avg=0.123    p50=0.120    p90=0.129    p99=0.229    max=0.229
```

## 3. 迭代时间对比 (measured 100 iters, excl warmup)

| 指标 | seq150+prefix20 | seq218+prefix25 | 增量 |
|------|----------------|----------------|------|
| 平均迭代时间 | **11,150 us** | **13,632 us** | +2,482 us (+22.3%) |
| 最小 | 11,080 us | 13,533 us | +2,453 us |
| 最大 | 11,996 us | 14,035 us | +2,039 us |
| 总时间 | 1,115 ms | 1,363 ms | +248 ms |

## 4. 设备端 Kernel 时间分解

### 4.1 按 Kernel Type 汇总

| Kernel Type | seq150+prefix20 (us) | 占比 | seq218+prefix25 (us) | 占比 | 变化 |
|------------|---------------------|------|---------------------|------|------|
| **KERNEL_AICORE (MatMul)** | 795,621 | **48.1%** | 1,025,580 | **50.6%** | +29.0% |
| **AI_VECTOR_CORE** | 424,537 | 25.7% | 533,397 | 26.3% | +25.6% |
| MIX_AIC (Attention) | 434,024 | 26.2% | 467,150 | 23.1% | +7.6% |
| **总计** | **1,654,182** | 100% | **2,026,127** | 100% | +22.5% |

### 4.2 每 Iteration (per-iter) 设备时间分解

| 类别 | seq150+prefix20 (us/iter) | 占比 | seq218+prefix25 (us/iter) | 占比 | 变化 |
|------|-------------------------|------|-------------------------|------|------|
| **MatMul - FFN Gate+Up (896→4864)** | 2,379 | 21.3% | 3,245 | 23.8% | +36.4% |
| **FusedInferAttentionScore** | 2,893 | 25.9% | 3,114 | 22.9% | +7.6% |
| **MatMul - FFN Down (4864→896)** | 1,320 | 11.8% | 1,833 | 13.5% | +38.9% |
| **GatherV2 (expand/compress)** | 742 | 6.7% | 750 | 5.5% | +1.1% |
| **InplaceAddRmsNorm** | 960 | 8.6% | 1,275 | 9.4% | +32.8% |
| **EltwiseBroadcastFusionOp** | 646 | 5.8% | 848 | 6.2% | +31.3% |
| MatMul - Q proj (896→896) | 563 | 5.0% | 550 | 4.0% | -2.3% |
| MatMul - O proj (896→896) | 525 | 4.7% | 621 | 4.6% | +18.3% |
| MatMul - K+V proj (896→128) | 508 | 4.6% | 579 | 4.3% | +13.9% |
| ApplyRotaryPosEmb | 444 | 4.0% | 643 | 4.7% | +44.8% |
| MatMul - lm_head (896→151936) | 9 | 0.1% | 8 | 0.1% | -11.1% |
| RmsNorm + Add | 39 | 0.4% | 41 | 0.3% | +5.1% |
| **总计** | **11,028** | 100% | **13,508** | 100% | +22.5% |

### 4.3 GatherV2 详细分解

| GatherV2 类型 | seq150+prefix20 | | seq218+prefix25 | |
|---------------|----------------|------|----------------|------|
| | count | avg (us) | count | avg (us) |
| Vocab embedding (151936→唯一) | 150 | 62.56 | 150 | 85.67 |
| Expand (唯一→全batch) | 3,750 | 13.28 | 3,750 | 14.17 |
| Compress (全batch→唯一) | 3,600 | 14.49 | 3,600 | 12.92 |
| **总计** | **7,500** | **14.85** | **7,500** | **15.00** |

每 iter GatherV2 开销: 742 us (seq150) / 750 us (seq218), 基本不随 seq 变化。

## 5. 与无 Prefix 对比

> 注意: 无 prefix 测试在 device 8 上运行, 有 prefix 在 device 4 上。两者硬件规格相同 (Ascend910_9382, 24 AIC, 48 AIV, 1800MHz), 但存在芯片级个体差异。

### 5.1 迭代时间对比

| 配置 | 设备 | 迭代时间 (us) | E2E (ms) | Execute (ms) |
|------|------|-------------|----------|-------------|
| seq150 无 prefix | 8 | 10,749 | - | - |
| seq150 +prefix20 | 4 | 11,150 | 11.732 | 11.200 |
| **差异** | | **+401 (+3.7%)** | | |
| seq218 无 prefix | 8 | 13,243 | - | - |
| seq218 +prefix25 | 4 | 13,632 | 14.410 | 13.683 |
| **差异** | | **+389 (+2.9%)** | | |

### 5.2 设备时间分解对比 (per-iter)

| 类别 | seq150 无prefix | seq150 +prefix20 | 差异 | seq218 无prefix | seq218 +prefix25 | 差异 |
|------|----------------|-----------------|------|----------------|-----------------|------|
| **KERNEL_AICORE (MatMul)** | 5,630 | 5,304 | **-326** | 7,455 | 6,837 | **-618** |
| **AI_VECTOR_CORE** | 2,241 | 2,830 | **+589** | 2,803 | 3,556 | **+753** |
| **MIX_AIC (Attention)** | 2,725 | 2,893 | **+168** | 2,814 | 3,114 | **+300** |
| **总计** | 10,596 | 11,028 | **+432** | 13,073 | 13,508 | **+435** |

### 5.3 MatMul Kernel 对比

| Kernel | seq150 无prefix avg | seq150 +prefix20 avg | 变化 | seq218 无prefix avg | seq218 +prefix25 avg | 变化 |
|--------|--------------------|--------------------|------|--------------------|--------------------|------|
| FFN Gate+Up (896→4864) | 54.54 | 49.56 | **-9.1%** | 75.74 | 67.61 | **-10.7%** |
| FFN Down (4864→896) | 58.77 | 55.01 | **-6.4%** | 82.00 | 76.37 | **-6.9%** |
| Q proj (896→896) | 11.64 | 11.73 | +0.8% | 12.58 | 11.47 | -8.8% |
| K+V proj (896→128) | 21.31 | 21.18 | -0.6% | 25.56 | 24.13 | -5.6% |
| O proj (896→896) | 21.78 | 21.86 | +0.4% | 26.09 | 25.87 | -0.8% |
| lm_head (896→151936) | 8.50 | 9.05 | +6.5% | 8.63 | 8.24 | -4.5% |

> FFN MatMul (Gate+Up + Down) 处理唯一 tokens (1320/1955), 随 prefix 去重而加速。
> Q/K/V/O proj 在 FusedInferAttentionScore 内部, 处理全 batch tokens (1500/2180), 基本不变。

### 5.4 增量来源拆解

seq150 +prefix20 vs 无prefix, 每 iter 变化:

| 来源 | 变化 (us/iter) | 说明 |
|------|---------------|------|
| **GatherV2 (expand/compress)** | **+672** | 50 次/iter × ~14us vs 2 次/iter × ~35us |
| FFN MatMul 节省 | -329 | Gate+Up: -239, Down: -90 (处理 1320 vs 1500 tokens) |
| Attention 变慢 | +168 | 全 batch 计算不变, 但 aic_total_cycles +9.4% (可能受 GatherV2 内存压力影响) |
| EltwiseBroadcastFusionOp | -67 | 处理唯一 tokens, 略有减少 |
| InplaceAddRmsNorm | -12 | 处理唯一 tokens, 略有减少 |
| **净增量** | **+432** | GatherV2 开销 > FFN 节省 |

seq218 +prefix25 vs 无prefix, 每 iter 变化:

| 来源 | 变化 (us/iter) | 说明 |
|------|---------------|------|
| **GatherV2 (expand/compress)** | **+662** | 50 次/iter × ~15us vs 2 次/iter × ~44us |
| FFN MatMul 节省 | -525 | Gate+Up: -390, Down: -135 (处理 1955 vs 2180 tokens) |
| Attention 变慢 | +300 | aic_total_cycles 增加约 7.3% |
| InplaceAddRmsNorm | +49 | 略有增加 |
| EltwiseBroadcastFusionOp | -65 | 处理唯一 tokens |
| **净增量** | **+421** | GatherV2 开销 > FFN 节省 |

## 6. Attention 算子详细对比

| 指标 | seq150 无prefix | seq150 +prefix20 | seq218 无prefix | seq218 +prefix25 |
|------|----------------|-----------------|----------------|-----------------|
| Input Shape (seq) | 1500,14,64 | 1500,14,64 | 2180,14,64 | 2180,14,64 |
| Task Duration (us) | 109.02 | 120.04 | 122.28 | 131.10 |
| aic_total_cycles | 4,396,536 | 4,809,693 | 4,922,817 | 5,271,452 |
| aic_mac_ratio | 3.2% | 2.9% | 4.2% | 4.0% |
| aic_mte2_ratio | 36.1% | 38.5% | 32.1% | 36.2% |
| aic_scalar_ratio | 48.2% | 45.9% | 49.3% | 47.2% |
| aiv_vec_ratio | 11.3% | 10.4% | 18.6% | 17.3% |
| cube_utilization | 93.35% | 92.75% | 93.19% | 93.08% |

> Attention input shape 相同 (全 batch), 但 aic_total_cycles 增加 7-9%。
> 可能原因: (1) device 4 vs 8 芯片差异; (2) GatherV2 内存操作导致 cache 竞争;
> (3) prefix KV cache 管理引入额外开销。

## 7. Host 侧 API 对比

| API | seq150 无prefix avg | seq150 +prefix20 avg | 变化 | seq218 无prefix avg | seq218 +prefix25 avg | 变化 |
|-----|--------------------|--------------------|------|--------------------|--------------------|------|
| aclmdlExecuteAsync | 5,319 | 6,232 | +17.1% | 5,448 | 5,220 | -4.2% |
| aclrtSynchronizeStream | 6,726 | 6,754 | +0.4% | 9,055 | 9,603 | +6.1% |
| MemCopySync | 85.4 | 155.2 | +81.7% | 93.3 | 66.2 | -29.0% |
| aclrtMemcpy count | 750 | 1,200 | +60.0% | 750 | 1,200 | +60.0% |
| KernelLaunchWithHandle count | 40,500 | 47,550 | +17.4% | 40,500 | 47,550 | +17.4% |

> KernelLaunchWithHandle 增加了 7,050 次 (= 50 GatherV2 × 150 iters - 2 原有 × 150 + ... 实际 = 7050 额外 launch), 与 GatherV2 增加一致。

## 8. 算子计数对比

| OP Type | 无prefix count | 有prefix count | 变化 | 说明 |
|---------|---------------|---------------|------|------|
| GatherV2 | 300 | 7,500 | +7,200 | 2→50 次/iter (expand/compress 每层 2 次) |
| InplaceAddRmsNorm | 7,050 | 7,050 | 0 | 不变 |
| FusedInferAttentionScore | 3,600 | 3,600 | 0 | 不变 |
| EltwiseBroadcastFusionOp | 3,600 | 3,600 | 0 | 不变 |
| ApplyRotaryPosEmb | 3,600 | 3,600 | 0 | 不变 |
| RmsNorm | 300 | 300 | 0 | 不变 |
| Add | 150 | 150 | 0 | 不变 |
| Sub | 150 | 0 | -150 | 有 prefix 时无 Sub 操作 |
| **总 launch** | 25,650 | 32,550 | +7,200 | 全部来自 GatherV2 |

## 9. 关键分析

### 9.1 Prefix Caching 当前为负优化

Prefix caching 在当前实现下**净增 ~430 us/iter (3-4%)**, 原因是 GatherV2 开销 (+662-672 us) 超过了 FFN 节省 (-329-525 us):

```
无 prefix:    FFN(1500 tokens) + Attention(1500 tokens)
有 prefix:    FFN(1320 tokens) + Attention(1500 tokens) + GatherV2(50次)

FFN 节省:     -329 us  (seq150)  /  -525 us  (seq218)
GatherV2 开销: +672 us  (seq150)  /  +662 us  (seq218)
Attention 开销: +168 us  (seq150)  /  +300 us  (seq218)
─────────────────────────────────────────────────
净增量:        +432 us  (seq150)  /  +435 us  (seq218)
```

### 9.2 FFN 节省随 prefix 比例增长

| 配置 | 去重率 | FFN 节省 (us/iter) | GatherV2 开销 (us/iter) | 净效果 |
|------|--------|-------------------|----------------------|--------|
| seq150, prefix20 | 12.0% | 329 | 672 | -343 (亏损) |
| seq218, prefix25 | 10.3% | 525 | 662 | -137 (亏损) |

> 去重率 = prefix_len × (batch-1) / (seq_len × batch)
> seq218 的 FFN 节省更大 (525 vs 329), 因为 prefix 虽然占比更低, 但绝对 token 数更多 (225 vs 180), 且单 token FFN 计算量更大 (seq 更长时 MatMul 更贵)。

### 9.3 理论 break-even 分析

假设 GatherV2 开销固定为 ~670 us/iter, Attention 开销增量与 GatherV2 次数成正比:

```
break-even: FFN_节省 = GatherV2_开销 + Attention_开销
            prefix_len × (batch-1) × f(seq_len) = 670 + 170

对于 seq=150: f(150) ≈ 329/180 = 1.83 us/token
  → 需要 prefix_len × 9 × 1.83 ≥ 840
  → prefix_len ≥ 51 (占 seq 的 34%)

对于 seq=218: f(218) ≈ 525/225 = 2.33 us/token
  → 需要 prefix_len × 9 × 2.33 ≥ 970
  → prefix_len ≥ 47 (占 seq 的 22%)
```

> 当前 prefix_len=20/25 远低于 break-even 点 (~47-51), 因此为负优化。

### 9.4 优化方向

1. **减少 GatherV2 次数**: 当前每层 2 次 (expand+compress), 可考虑:
   - 在 FFN 中直接处理全 batch tokens 但跳过 prefix 行 (masked FFN)
   - 或将 expand/compress 融入相邻算子 (如 RmsNorm+Gather 融合)

2. **减少 GatherV2 单次开销**: 当前 ~14 us/call, 可通过:
   - 优化 gather kernel 实现 (如使用 vector core 并行)
   - 减小 gather 数据量 (只搬运 hidden_state 而非完整 tensor)

3. **增大 prefix_len**: 当前 20-25 过小, 增至 50+ 可达到 break-even

4. **Attention 层不 expand/compress**: 当前 attention 处理全 batch (1500/2180), 其 Q/K/V/O proj 也处理全 batch。如果 attention 也能利用 prefix 去重 (如 prefix KV cache), 则可进一步节省。

## 10. 结论

1. **Prefix caching 当前实现为负优化**, 净增 ~430 us/iter (3-4%), 主要因 GatherV2 expand/compress 开销 (+670 us) 超过 FFN 节省 (-330~525 us)
2. **FFN MatMul 确实受益**, Gate+Up 加速 9-11%, Down 加速 6-7%, 但节省量不足以覆盖 gather 开销
3. **Attention 不受益**, 仍处理全 batch tokens, 且可能因 GatherV2 内存竞争而变慢 6-11%
4. **Break-even prefix_len 约为 47-51** (当前 seq 的 22-34%), 远高于测试的 20-25
5. 优化建议: 减少 GatherV2 次数/开销, 或增大 prefix_len, 或让 Attention 也利用 prefix 去重

## 11. 附录 A: GatherV2 Shape 验证

| 配置 | Vocab Gather | Expand | Compress | 验证 |
|------|-------------|--------|----------|------|
| seq150, prefix20 | 151936→1320 | 1320→1500 | 1500→1320 | 1320 = 20+(150-20)×10 ✓ |
| seq218, prefix25 | 151936→1955 | 1955→2180 | 2180→1955 | 1955 = 25+(218-25)×10 ✓ |

## 12. 附录 B: 设备信息

| 项目 | 值 |
|------|-----|
| SoC | Ascend910_9382 |
| Device ID | 4 |
| AI Core 数 | 24 |
| AI Vector Core 数 | 48 |
| AIC/AIV 频率 | 1800 MHz |
| 驱动版本 | 467992 |
| Hostname | A03-R40-I191-99-4100006.JD.LOCAL |
| pid | 830477 (seq150) / 830700 (seq218) |
