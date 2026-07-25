#!/usr/bin/env python3
"""Plot QPS and E2E latency (avg/p99) vs threads for bench_latency sweep."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

threads   = [1, 2, 3, 4, 5, 6, 7, 8]
qps       = [78.70, 91.02, 95.75, 98.53, 99.67, 99.54, 99.18, 98.67]
tps       = [117995, 136414, 143508, 147623, 149325, 149237, 148685, 147921]
e2e_avg   = [12.70, 21.94, 31.32, 40.57, 50.13, 60.24, 70.54, 81.05]
e2e_p99   = [14.30, 26.92, 34.65, 44.99, 54.67, 64.84, 75.07, 85.25]

fig, ax1 = plt.subplots(figsize=(10, 6))

color_qps = '#2196F3'
color_tps = '#4CAF50'
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

ax1.axvline(x=5, color='gray', alpha=0.5, linestyle='-.', linewidth=1)
ax1.annotate('Optimal\n(5 threads)', xy=(5, max(qps)*1.15), fontsize=9,
             color='gray', ha='center')

plt.title('Qwen2.5-0.5B: QPS & Latency vs Threads (batch=10, seq~150, Ascend910_9382)',
          fontsize=13, pad=12)
plt.tight_layout()

out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   'models', 'qwen2.5-0.5b', 'docs', 'latency_qps_curve.png')
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f'Saved: {out}')
