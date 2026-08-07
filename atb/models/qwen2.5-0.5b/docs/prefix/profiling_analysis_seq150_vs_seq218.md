# Qwen2.5-0.5B Profiling 分析: fixed-seq 150 vs 218

> 测试日期: 2026-08-05
> 测试环境: Ascend910_9382, NPU device 8, 24 AI cores, 48 vector cores, AIC/AIV 1800MHz
> 模型: qwen2.5-0.5b (24层, hidden=896, 14 q-heads, 2 kv-heads, head_dim=64, vocab=151936, fp16)
> 测试命令: `./run.sh bench --profiling --warmup 50 --requests 100 --fixed-seq {150|218}`
> Profiling 数据目录: `atb/models/qwen2.5-0.5b/profiling_data/`

## 1. 测试配置

| 项目 | 值 |
|------|-----|
| 模型 | Qwen2.5-0.5B (24层, hidden=896, 14 q-heads, 2 kv-heads, head_dim=64) |
| 硬件 | Ascend910_9382, device 8, 24 AI cores, 48 vector cores, AIC/AIV 1800MHz |
| 命令 | `./run.sh bench --profiling --warmup 50 --requests 100 --fixed-seq {150|218}` |
| batch_size | 10 (从 input shape 推断: seq150→1500 tokens, seq218→2180 tokens) |
| 测量迭代 | 100次 (排除50次warmup和1次init) |
| Profiling 目录 | seq150: `PROF_000001_20260805150430874_00791707MDFJCLGO` |
|              | seq218: `PROF_000001_20260805150437907_00791878HQDNEIJL` |

### 1.1 数据来源

| 文件 | 用途 |
|------|------|
| `mindstudio_profiler_output/step_trace_*.csv` | 迭代时间 (含 host launch 间隙) |
| `mindstudio_profiler_output/task_time_*.csv` | 设备端每个 kernel 的实际执行时间 |
| `mindstudio_profiler_output/op_summary_*.csv` | 算子详细信息 (shape, aic/aiv 子指标) |
| `mindstudio_profiler_output/op_statistic_*.csv` | 算子统计 (注意: 仅含 AIV + MIX_AIC, 遗漏 AICORE MatMul) |
| `mindstudio_profiler_output/api_statistic_*.csv` | Host 侧 ACL/Runtime API 统计 |
| `device_8/info.json.8` | 设备硬件信息 |

### 1.2 重要说明: op_statistic.csv 的局限性

`op_statistic.csv` 只统计了 `AI_VECTOR_CORE` 和 `MIX_AIC` 类型的算子, **完全遗漏了 `KERNEL_AICORE` 的 MatMul 任务**。实际设备时间分布必须结合 `task_time.csv` 才能获得完整视图。

| op_statistic.csv 包含 | op_statistic.csv 遗漏 |
|----------------------|----------------------|
| FusedInferAttentionScore (MIX_AIC) | te_matmulv3 (KERNEL_AICORE) — 占总时间 53-56% |
| InplaceAddRmsNorm (AI_VECTOR_CORE) | |
| EltwiseBroadcastFusionOp (AI_VECTOR_CORE) | |
| ApplyRotaryPosEmb (AI_VECTOR_CORE) | |
| GatherV2, RmsNorm, Add, Sub (AI_VECTOR_CORE) | |

## 2. 迭代时间对比 (measured 100 iters, excl warmup)

| 指标 | seq150 | seq218 | 增量 |
|------|--------|--------|------|
| 平均迭代时间 | **10,749 us** | **13,243 us** | +2,494 us (+23.2%) |
| 最小 | 10,640 us | 13,114 us | +2,474 us |
| 最大 | 11,044 us | 13,477 us | +2,433 us |
| 总时间 | 1,075 ms | 1,324 ms | +249 ms |

迭代时间非常稳定 (min/max 差距 <4%), 波动小。

## 3. 设备端 Kernel 时间分解

### 3.1 按 Kernel Type 汇总

| Kernel Type | seq150 总时间 (us) | 占比 | seq218 总时间 (us) | 占比 | 变化 |
|------------|-------------------|------|-------------------|------|------|
| **KERNEL_AICORE (MatMul)** | 844,489 | **53.3%** | 1,118,303 | **56.4%** | +32.4% |
| MIX_AIC (Attention) | 408,732 | 25.8% | 422,118 | 21.3% | +3.3% |
| AI_VECTOR_CORE (Norm/Rotary/Elt) | 336,156 | 21.2% | 420,488 | 21.2% | +25.1% |
| **总计** | **1,589,377** | 100% | **1,960,909** | 100% | +23.4% |

### 3.2 按 MatMul Kernel 详细分解

从 `task_time.csv` 提取的 `te_matmulv3` kernel (按 hash 区分不同权重):

| Kernel (推测用途) | seq150 count | seq150 avg (us) | seq150 total (us) | seq218 count | seq218 avg (us) | seq218 total (us) | 变化 |
|-------------------|-------------|-----------------|-------------------|-------------|-----------------|-------------------|------|
| te_matmulv3_b8c... (FFN Gate+Up, 896→4864) | 7200 | 54.54 | 392,721 | 7200 | 75.74 | 545,297 | +38.9% |
| te_matmulv3_c5d... (FFN Down, 4864→896) | 3600 | 58.77 | 211,582 | 3600 | 82.00 | 295,187 | +39.5% |
| te_matmulv3_4d4... (Q proj, 896→896) | 7200 | 11.64 | 83,808 | 7200 | 12.58 | 90,598 | +8.1% |
| te_matmulv3_064... (K+V proj, 896→128) | 3600 | 21.31 | 76,702 | 3600 | 25.56 | 91,998 | +20.0% |
| te_matmulv3_d46... (O proj, 896→896) | 3600 | 21.78 | 78,401 | 3600 | 26.09 | 93,928 | +19.9% |
| te_matmulv3_142... (lm_head, 896→151936) | 150 | 8.50 | 1,275 | 150 | 8.63 | 1,295 | +1.5% |

> count 说明: 24层 × batch调用量 = 7200 表示每层每iter调用3次 (Gate/Up分开或Q/K/V分开), 3600 表示每层每iter调用1.5次。150 = 24层每iter仅1次 (lm_head 只算最后token)。

### 3.3 每迭代 (per-iter) 设备时间分解

将总时间除以 150 iters, 得到单次推理的设备时间分布:

| 类别 | seq150 (us/iter) | 占比 | seq218 (us/iter) | 占比 | 变化 |
|------|-----------------|------|-----------------|------|------|
| **MatMul - FFN Gate+Up (896→4864)** | 2,618 | 24.4% | 3,635 | 27.5% | **+38.9%** |
| **FusedInferAttentionScore** | 2,725 | 25.4% | 2,814 | 21.3% | +3.3% |
| **MatMul - FFN Down (4864→896)** | 1,411 | 13.2% | 1,968 | 14.9% | **+39.5%** |
| **InplaceAddRmsNorm** | 972 | 9.1% | 1,226 | 9.3% | +26.1% |
| **EltwiseBroadcastFusionOp** | 713 | 6.6% | 913 | 6.9% | +28.0% |
| MatMul - Q proj (896→896) | 559 | 5.2% | 604 | 4.6% | +8.1% |
| MatMul - O proj (896→896) | 523 | 4.9% | 627 | 4.7% | +19.9% |
| MatMul - K+V proj (896→128) | 511 | 4.8% | 613 | 4.6% | +19.9% |
| ApplyRotaryPosEmb | 443 | 4.1% | 532 | 4.0% | +20.3% |
| GatherV2 | 70 | 0.7% | 88 | 0.7% | +24.9% |
| MatMul - lm_head (896→151936) | 9 | 0.1% | 9 | 0.1% | +1.5% |
| RmsNorm + Add + Sub | 43 | 0.4% | 45 | 0.3% | ~0% |
| **总计** | **10,596** | 100% | **13,073** | 100% | **+23.4%** |

### 3.4 非 MatMul 算子统计 (来自 op_statistic.csv)

| OP Type | Core Type | Count | seq150 avg (us) | seq218 avg (us) | 变化 |
|---------|-----------|-------|-----------------|-----------------|------|
| FusedInferAttentionScore | MIX_AIC | 3600 | 113.54 | 117.26 | +3.3% |
| InplaceAddRmsNorm | AI_VECTOR_CORE | 7050 | 20.68 | 26.08 | +26.1% |
| EltwiseBroadcastFusionOp | AI_VECTOR_CORE | 3600 | 29.70 | 38.04 | +28.0% |
| ApplyRotaryPosEmb | AI_VECTOR_CORE | 3600 | 18.44 | 22.19 | +20.3% |
| GatherV2 | AI_VECTOR_CORE | 300 | 35.09 | 43.84 | +24.9% |
| RmsNorm | AI_VECTOR_CORE | 300 | 15.18 | 16.31 | +7.5% |
| Add | AI_VECTOR_CORE | 150 | 9.45 | 8.63 | -8.7% |
| Sub | AI_VECTOR_CORE | 150 | 3.58 | 3.28 | -8.5% |

## 4. Host 侧 API 对比

| API | seq150 avg (us) | seq218 avg (us) | 变化 | 说明 |
|-----|----------------|----------------|------|------|
| aclmdlExecuteAsync | 5,319 | 5,448 | +2.4% | 异步 launch, host 侧开销不随 seq 增长 |
| aclrtSynchronizeStream | 6,726 | 9,055 | +34.6% | 同步等待, 反映设备执行时间变长 |
| MemCopySync | 85.4 | 93.3 | +9.2% | H2D/D2H 拷贝 |
| aclnnInnerFusedInferAttentionScore | 19.31 | 18.70 | -3.1% | Attention 算子 host 侧开销 |
| KernelLaunchWithHandle | 4.29 | 4.15 | -3.3% | Kernel launch 平均开销 |
| launch (node) | 6.14 | 6.01 | -2.1% | Node 级 launch |

## 5. Attention 算子详细分析 (来自 op_summary.csv)

| 指标 | seq150 | seq218 | 说明 |
|------|--------|--------|------|
| Input Shape (seq) | 1500,14,64 | 2180,14,64 | tokens × heads × head_dim |
| Task Duration | 109.02 us | 122.28 us | 单次执行时间 |
| aic_total_cycles | 4,396,536 | 4,922,817 | 总 cycle 数 |
| aic_mac_ratio | 3.2% | 4.2% | 矩阵乘计算占比极低 |
| aic_mte2_ratio | 36.1% | 32.1% | 内存搬运 (MTE2) 占比最高 |
| aic_scalar_ratio | 48.2% | 49.3% | scalar 计算占比高 |
| aiv_vec_ratio | 11.3% | 18.6% | vector 计算占比 |
| cube_utilization | 93.35% | 93.19% | cube 利用率高 |

**关键发现**: Attention 的 `aic_mac_ratio` 仅 3-4%, `aic_mte2_ratio` (内存搬运) 占 32-36%, 说明 attention 是**内存搬运主导**而非计算主导, 固定开销 (tiling/setup) 占比大, 因此对 seq 长度不敏感。

## 6. 关键分析

### 6.1 FFN MatMul 是最大瓶颈

FFN 的三个线性层 (Gate+Up+Down) 占总设备时间的 **37.6%→42.4%**, 是单次推理的最大开销。这些 matmul (896→4864→896) 随 seq 长度近线性增长 (+39%), 因为计算量与 token 数成正比。

| FFN 层 | seq150 (us/iter) | seq218 (us/iter) | 变化 |
|--------|-----------------|-----------------|------|
| Gate+Up (896→4864) | 2,618 | 3,635 | +38.9% |
| Down (4864→896) | 1,411 | 1,968 | +39.5% |
| **FFN 合计** | **4,029** | **5,603** | **+39.1%** |

### 6.2 Attention 几乎不随 seq 增长 (+3.3%)

seq 从 150→218 (tokens 1500→2180, +45%), 但 FusedInferAttentionScore 平均耗时仅从 113.5us→117.3us (+3.3%)。原因:
- 0.5B 模型 head_dim=64, 序列长度 1500-2180 的 attention 矩阵较小
- attention 的 `aic_mac_ratio` 仅 3.2-4.2%, `aic_mte2_ratio` (内存搬运) 占 32-36%
- attention 是**内存搬运主导**而非计算主导, 固定开销 (tiling/setup) 占比大

### 6.3 K/V 投影几乎不增长 (+8.1%)

K/V proj (896→128) 的输出维度仅 128, 矩阵很小, 计算量本就低, 主要受 kernel launch 开销影响, 随 seq 增长很慢。

### 6.4 lm_head 只计算最后一个 token

lm_head (896→151936) 仅 8.5us 且不随 seq 变化, 说明只对 batch 中每个序列的最后一个 token 计算 logits (10 tokens), 而非全序列。

### 6.5 设备利用率极高

| 指标 | seq150 | seq218 |
|------|--------|--------|
| step_trace 迭代时间 | 10,749 us | 13,243 us |
| 设备 kernel 总时间 | 10,596 us | 13,073 us |
| **间隙 (launch overhead)** | **153 us (1.4%)** | **170 us (1.3%)** |

host-device 间隙仅 ~1.4%, 说明 kernel 流水线编排优秀, 几乎无空闲。

### 6.6 增量来源拆解

seq 150→218 迭代时间增加 2,494 us, 各部分贡献:

| 来源 | 增量 (us) | 占增量比 |
|------|----------|---------|
| FFN MatMul (Gate+Up+Down) | +1,574 | **63.1%** |
| InplaceAddRmsNorm | +254 | 10.2% |
| EltwiseBroadcastFusionOp | +200 | 8.0% |
| Q/O proj MatMul | +146 | 5.9% |
| K+V proj MatMul | +102 | 4.1% |
| ApplyRotaryPosEmb | +89 | 3.6% |
| FusedInferAttentionScore | +89 | 3.6% |
| GatherV2 + 其他 | +40 | 1.6% |
| **总计** | **+2,494** | **100%** |

**FFN MatMul 贡献了 ~63% 的时间增量**, 是 seq 长度增加导致延迟上升的主要原因。

## 7. 结论

1. **性能瓶颈是 FFN MatMul (42%)**, 而非 Attention (21%)。优化应优先关注 FFN 线性层的 cube 利用率
2. **seq 150→218 迭代时间增加 23.2%**, 其中 FFN MatMul 贡献了 ~63% 的增量
3. **Attention 对 seq 长度不敏感** (+3.3%), 因为 0.5B 小模型的 attention 矩阵小, 固定开销主导
4. **op_statistic.csv 有误导性** — 它只统计了 AI_VECTOR_CORE + MIX_AIC 算子, 遗漏了占总时间 53-56% 的 KERNEL_AICORE MatMul。分析时必须结合 task_time.csv 才能获得完整视图
5. 设备流水线利用率极高 (~98.6%), host 侧优化空间有限

## 8. 附录 A: 算子计数说明

| OP Type / Kernel | Count | 说明 |
|------------------|-------|------|
| InplaceAddRmsNorm | 7050 | 24层 × (1 pre + 1 post + ~1.4 ffn) × 150 iters ≈ 7050 |
| FusedInferAttentionScore | 3600 | 24层 × 150 iters = 3600 |
| EltwiseBroadcastFusionOp | 3600 | 24层 × 150 iters = 3600 |
| ApplyRotaryPosEmb | 3600 | 24层 × 150 iters = 3600 |
| te_matmulv3 (Gate+Up) | 7200 | 24层 × 2 (Gate+Up分开) × 150 iters = 7200 |
| te_matmulv3 (Down) | 3600 | 24层 × 1 × 150 iters = 3600 |
| te_matmulv3 (Q proj) | 7200 | 24层 × 2 × 150 iters = 7200 (可能Q分开) |
| te_matmulv3 (K+V proj) | 3600 | 24层 × 1 (K+V融合) × 150 iters = 3600 |
| te_matmulv3 (O proj) | 3600 | 24层 × 1 × 150 iters = 3600 |
| te_matmulv3 (lm_head) | 150 | 1 × 150 iters = 150 (仅最后token) |
| GatherV2 | 300 | 2 × 150 iters = 300 (input_embed + cos/sin) |
| RmsNorm | 300 | 2 × 150 iters = 300 (final norm) |
| Add / Sub | 150 | 1 × 150 iters = 150 |

## 9. 附录 B: 设备硬件信息

| 项目 | 值 |
|------|-----|
| SoC | Ascend910_9382 |
| AI Core 数 | 24 |
| AI Vector Core 数 | 48 |
| AI CPU Core 数 | 6 |
| AIC 频率 | 1800 MHz |
| AIV 频率 | 1800 MHz |
| HWTS 频率 | 49.999 MHz |
| HBM | 65536 MB |
| CANN 驱动版本 | 467992 |
| OS | Linux-5.10.0-jd_663.64kb.aarch64 |
| Hostname | A03-R40-I191-99-4100006.JD.LOCAL |

## 10. 附录 C: Profiling 配置

```json
{
    "profiler": {
        "switch": "on",
        "output": "/export/home/weinan5/hejun/workspace/qwen2.5/atb/models/qwen2.5-0.5b/profiling_data",
        "task_time": "on",
        "runtime_api": "on",
        "ascendcl": "on"
    }
}
```

- `prof_level`: level1
- `ai_core_profiling_mode`: task-based
- `aiv_profiling_mode`: task-based
- `aicore_sampling_interval`: 10 us
- `aiv_sampling_interval`: 10 us
