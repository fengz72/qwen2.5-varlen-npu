#!/usr/bin/env python3
"""Plot QPS and E2E latency (avg/p99) vs threads for bench_latency sweep."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

threads   = [1, 2, 3, 4, 5, 6, 7, 8]
qps       = [67.30, 79.31, 79.73, 80.01, 80.00, 79.46, 79.18, 79.07]
e2e_avg   = [14.86, 25.21, 37.60, 49.96, 62.47, 75.46, 88.33, 101.11]
e2e_p99   = [16.17, 27.09, 40.88, 54.52, 68.79, 83.33, 96.58, 108.40]

fig, ax1 = plt.subplots(figsize=(10, 6))

color_qps = '#2196F3'
color_avg = '#F44336'
color_p99 = '#FF9800'

ax1.set_xlabel('Threads', fontsize=13)
ax1.set_ylabel('QPS (req/s)', color=color_qps, fontsize=13)
ln1 = ax1.plot(threads, qps, 'o-', color=color_qps, linewidth=2, markersize=7, label='QPS')
ax1.tick_params(axis='y', labelcolor=color_qps)
ax1.set_ylim(0, max(qps) * 1.25)
ax1.set_xticks(threads)

ax2 = ax1.twinx()
ax2.set_ylabel('E2E Latency (ms)', fontsize=13)
ln2 = ax2.plot(threads, e2e_avg, 's--', color=color_avg, linewidth=2, markersize=7, label='E2E avg')
ln3 = ax2.plot(threads, e2e_p99, '^:', color=color_p99, linewidth=2, markersize=7, label='E2E p99')
ax2.tick_params(axis='y')
ax2.set_ylim(0, max(e2e_p99) * 1.15)

lines = ln1 + ln2 + ln3
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='center right', fontsize=11)

ax1.grid(True, alpha=0.3, linestyle='--')

ax1.axvline(x=2, color='gray', alpha=0.5, linestyle='-.', linewidth=1)
ax1.annotate('Saturation\npoint', xy=(2, max(qps)*1.15), fontsize=9,
             color='gray', ha='center')

plt.title('Qwen2.5-0.5B: QPS & Latency vs Threads (Independent Threads)',
          fontsize=14, pad=12)
plt.tight_layout()

out = '/export/home/weinan5/hejun/workspace/qwen2.5/atb/models/qwen2.5-0.5b/docs/latency_qps_curve.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f'Saved: {out}')
