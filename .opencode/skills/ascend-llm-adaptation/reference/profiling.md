# Profiling

## 执行 profiling

用 CANN 自带 `msprof` 工具对 OM 推理做 profiling（单线程 + 固定 seq，保证对比公平）：

```bash
# 通过 acl inference 触发 profiling，产出 profiling_data/ 目录
# 需在推理前设置环境变量
export PROFILING_MODE="enable"
export PROFILING_OPTIONS="{'output':'./profiling_data','training':'off','aclnn':'on','l2':'on','op_summary':'on'}"

# 执行推理（用任意 ACL 推理脚本/ais_bench 触发一次推理即可）
# 推理结束后 profiling_data/ 下生成 csv 汇总文件
```

或用 `torch_npu` 的 Python profiling API（适用于 eager / 导出阶段验证）：

```python
import torch_npu
with torch_npu.npu.profile(profiling_result_path="./profiling_data"):
    model(input_ids, ...)
```

## 分析流程

1. 读取 `op_summary_*.csv`，按 `aclnn` 执行时间识别 top-K 算子
2. 检查 `aic_mac_ratio` — 低于 60% 说明 tiling 有问题
3. 检查 `aic_mte2_ratio` — 高说明访存瓶颈
4. 对可疑算子对比融合 vs 拆分方案
5. 在 Step 2 调整融合策略
6. 重新 export AIR + 编译 OM + benchmark 验证

## 关键指标

| 指标 | 良好范围 | 含义 |
|------|---------|------|
| `aic_mac_ratio` | >70% | Cube MAC 利用率 |
| `aic_mte2_ratio` | <80% | HBM 数据加载占比 |
| `cube_utilization` | >85% | Cube 整体忙碌率 |
| Execute avg | baseline | NPU 执行时间 |
| QPS | baseline | 每秒请求数 |
| Token 吞吐 | baseline | 每秒 token 数 |

## 迭代示例（Qwen2.5-0.5b）

profiling 发现 `npu_ffn`（FFNV3）占执行时间 54%，MAC 仅 43.5%。

根因：两个不同 MatMul shape（K=896 vs K=4864）的 tiling 妥协。

去掉 FFN 融合 → 拆分为独立 MatMul → MAC 85% → MLP 快 16.4%。

```
Step 6 profiling → 发现 FFN MAC 43%
  → 回 Step 2: 去掉 npu_ffn 融合
  → 重走 Step 4: 重新 export + ATC + OM
  → Step 6 再测: MAC 85%, MLP 快 16.4%
```

### FFN 融合 vs 拆分对比数据

> 测试环境: Ascend910_9382, batch=10, seq_len=208, 单线程

| 指标 | 融合 FFN | 不融合 FFN | 提升 |
|------|----------|------------|------|
| Execute avg | 16.706 ms | 15.487 ms | -7.3% |
| QPS | 56.68 | 61.92 | +9.3% |
| Token 吞吐 | 117,887 | 128,786 | +9.3% |
| MLP MAC ratio | 43.5% | 85% | +95% |
| MLP 总耗时/iter | 9,276 us | 7,754 us | -16.4% |
| 全算子总耗时/iter | 17,188 us | 15,302 us | -11.0% |
