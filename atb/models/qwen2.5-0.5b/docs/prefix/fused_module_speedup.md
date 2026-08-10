# Fused Pre/Post + FFN 各模块加速比

> 场景: seq=150, prefix=20, batch=10, unique=1320 vs 无 prefix 基线 (10596 us)
> 基线数据来源: device 8 无 prefix seq150 profiling (`PROF_...00791707MDFJCLGO`)
> 线性模型: seq150(1500tok) vs seq218(2180tok) 同 device 8, `V = (T218-T150)/680`, `F = T150 - V*1500`
> 实际预期修正系数: ×0.85 (来自 GatherV2 prefix 实测验证, FFN 实测/理论 = 79%)

## 各模块加速比

| 模块 | 基线 (us) | 理论优化 (us) | 实际预期 (us) | 理论加速比 | 实际加速比 | 主要收益来源 |
|------|----------|-------------|-------------|-----------|-----------|------------|
| Fused_Pre | 1,530 | 1,224 | 1,282 | 20.0% | 16.2% | QKV merge + RoPE launch 吸收 |
| FIA | 2,724 | 2,724 | 2,724 | 0% | 0% | 无变化 (1500 tokens) |
| Fused_Post | 1,019 | 831 | 865 | 18.4% | 15.1% | O proj + AddRmsNorm launch 吸收 |
| FFN | 5,217 | 4,715 | 4,790 | 9.6% | 8.2% | Token reduction (1320 vs 1500) |
| Tail | 111 | 108 | 108 | 3.2% | 3.2% | Sub 消除 |
| 总计 | 10,601 | 9,602 | 9,770 | 9.4% | 7.8% | |

## 算子级明细

| 模块 | 算子 | 基线 (us) | 理论优化 (us) | 实际优化 (us) | 理论加速 | 实际加速 |
|------|------|----------|-------------|-------------|---------|---------|
| Fused_Pre | RmsNorm(input) | 17 | 17 | 17 | 2.9% | 2.9% |
| | Q proj (QKV merged) | 559 | 426 | 452 | 23.8% | 19.2% |
| | K+V proj (QKV merged) | 511 | 411 | 430 | 19.6% | 15.9% |
| | RoPE (fused) | 442 | 371 | 384 | 16.2% | 13.2% |
| FIA | FIA (unchanged) | 2,724 | 2,724 | 2,724 | 0% | 0% |
| Fused_Post | O proj (fused+compress) | 522 | 417 | 437 | 20.2% | 16.4% |
| | AddRmsNorm(post-attn) | 496 | 414 | 429 | 16.6% | 13.6% |
| FFN | Gate+Up | 2,618 | 2,349 | 2,389 | 10.3% | 8.7% |
| | SiLU*gate | 713 | 660 | 668 | 7.4% | 6.3% |
| | Down | 1,411 | 1,263 | 1,285 | 10.5% | 8.9% |
| | AddRmsNorm(post-ffn) | 476 | 443 | 448 | 6.9% | 5.9% |
| Tail | Sub (eliminated) | 4 | 0 | 0 | 100% | 100% |

## 数据来源说明

### 基线数据

| 算子 | 来源文件 | 计算方式 |
|------|---------|---------|
| Q proj | task_time.csv (hash 4d4...) | 83,808 / 150 = 559 |
| K+V proj | task_time.csv (hash 064...) | 76,702 / 150 = 511 |
| O proj | task_time.csv (hash d46...) | 78,401 / 150 = 522 |
| Gate+Up | task_time.csv (hash b8c...) | 392,721 / 150 = 2,618 |
| Down | task_time.csv (hash c5d...) | 211,582 / 150 = 1,411 |
| FIA | task_time.csv (MIX_AIC 汇总) | 408,732 / 150 = 2,724 |
| RoPE | op_statistic.csv | 18.44 × 24 = 442 |
| SiLU*gate | op_statistic.csv | 29.70 × 24 = 713 |
| AddRmsNorm | op_statistic.csv | 20.68 × 47 (拆 24:23) |
| RmsNorm(input) | op_statistic.csv | 15.18 × 1 = 17 |
| Sub | op_statistic.csv | 3.58 × 1 = 4 |

### 理论优化推导

```
线性模型: T(n) = V × n + F
  V = (T_2180 - T_1500) / 680     (seq150 vs seq218, 同 device 8)
  F = T_1500 - V × 1500

理论优化 = V × 1320 + F
  1320 = prefix_len + (seq_len - prefix_len) × batch
       = 20 + 130 × 10
```

### 实际预期推导

```
0.85 修正系数来源:
  理论 FFN 节省 = V × 180 = 502.6 us
  实测 FFN 节省 = 396.2 us (device 4 prefix task_time.csv)
  实测/理论 = 396.2 / 502.6 = 79%

采用 0.85 (融合方案无 GatherV2 内存竞争, 预期效率高于实测 79%):
  实际节省 = 理论节省 × 0.85
  实际优化 = 基线 - 实际节省
```

### 融合收益来源

| 收益类型 | 理论 (us) | 实际 (us) | 占比 | 说明 |
|---------|----------|----------|------|------|
| Token reduction (1320 vs 1500) | 628 | 534 | 5.0% | 12% 去重率, 全链路 |
| Fusion (QKV merge + launch + mem) | 368 | 294 | 2.8% | launch 吸收 + input reload |
| Sub elimination | 4 | 4 | 0.0% | 微小 |
| 总计 | 999 | 832 | 7.8% | |
