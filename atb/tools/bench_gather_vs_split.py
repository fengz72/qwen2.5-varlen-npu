#!/usr/bin/env python3
"""GatherV2 vs SplitV+ConcatV2 单算子性能对比测试。

测试 prefix sharing 场景下 expand/restore 两种实现的单算子耗时:
  - 方案 A: torch.index_select (GatherV2, 逐 token 散列读)
  - 方案 B: torch.split + torch.cat (SplitV+ConcatV2, 连续 DMA)

测试参数对齐线上分布: avg_seq=150, p99=218, prefix=20-25。

用法:
    python -m atb.tools.bench_gather_vs_split --device 8
    python -m atb.tools.bench_gather_vs_split --device 8 --n-list 1,3,5,8,10
"""

import argparse
import random
import time

import numpy as np
import torch
import torch_npu


def build_expand_index(prefix_len, seq_lens, device):
    indices = []
    compact_req_offset = prefix_len
    for sl in seq_lens:
        req_tokens = sl - prefix_len
        indices.append(torch.arange(prefix_len, device=device))
        indices.append(torch.arange(compact_req_offset, compact_req_offset + req_tokens, device=device))
        compact_req_offset += req_tokens
    return torch.cat(indices).to(torch.int64)


def build_restore_index(prefix_len, seq_lens, device):
    prefix_idx = torch.arange(prefix_len, device=device)
    req_indices = []
    block_start = 0
    for sl in seq_lens:
        req_tokens = sl - prefix_len
        req_start = block_start + prefix_len
        req_indices.append(torch.arange(req_start, req_start + req_tokens, device=device))
        block_start += sl
    return torch.cat([prefix_idx] + req_indices).to(torch.int64)


def expand_gather(hidden, expand_index):
    return torch.index_select(hidden, 0, expand_index)


def restore_gather(expanded, restore_index):
    return torch.index_select(expanded, 0, restore_index)


def expand_split(hidden, prefix_len, num_reqs, req_lens):
    D = hidden.shape[-1]
    prefix = hidden[:prefix_len]
    prefix_3d = prefix.unsqueeze(0).expand(num_reqs, prefix_len, D)
    reqs = hidden[prefix_len:]
    req_splits = torch.split(reqs, req_lens, dim=0)
    parts = []
    for i in range(num_reqs):
        parts.append(prefix_3d[i])
        parts.append(req_splits[i])
    return torch.cat(parts, dim=0)


def restore_split(expanded, prefix_len, num_reqs, block_lens):
    block_splits = torch.split(expanded, block_lens, dim=0)
    prefix = block_splits[0][:prefix_len]
    req_parts = [blk[prefix_len:] for blk in block_splits]
    return torch.cat([prefix] + req_parts, dim=0)


def generate_seq_lens(num_reqs, avg=150, p99=218, seed=None):
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()

    sigma = 0.15
    mu = np.log(avg) - sigma ** 2 / 2
    seqs = rng.lognormal(mean=mu, sigma=sigma, size=num_reqs)
    seqs = np.clip(seqs, 50, p99).astype(int)
    if num_reqs > 0 and seqs.max() < p99:
        idx = rng.randint(num_reqs)
        seqs[idx] = p99
    return seqs.tolist()


def benchmark_fn(fn, args, warmup=50, iters=1000):
    for _ in range(warmup):
        fn(*args)
    torch.npu.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    torch.npu.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) / iters * 1e6


def run_single_config(N, D, dtype, device, seed=42):
    seq_lens = generate_seq_lens(N, avg=150, p99=218, seed=seed)
    prefix_len = random.Random(seed).randint(20, 26)
    req_lens = [sl - prefix_len for sl in seq_lens]
    block_lens = [sl for sl in seq_lens]

    T_compact = prefix_len + sum(req_lens)
    T_expanded = sum(seq_lens)

    hidden = torch.randn(T_compact, D, dtype=dtype, device=device)

    expand_index = build_expand_index(prefix_len, seq_lens, device)
    restore_index = build_restore_index(prefix_len, seq_lens, device)

    expanded_gather = expand_gather(hidden, expand_index)
    expanded_split = expand_split(hidden, prefix_len, N, req_lens)
    correct_expand = torch.allclose(expanded_gather, expanded_split)

    restored_gather = restore_gather(expanded_gather, restore_index)
    restored_split = restore_split(expanded_gather, prefix_len, N, block_lens)
    correct_restore = torch.allclose(restored_gather, restored_split)

    expand_gather_us = benchmark_fn(expand_gather, (hidden, expand_index))
    restore_gather_us = benchmark_fn(restore_gather, (expanded_gather, restore_index))

    expand_split_us = benchmark_fn(expand_split, (hidden, prefix_len, N, req_lens))
    restore_split_us = benchmark_fn(restore_split, (expanded_gather, prefix_len, N, block_lens))

    total_gather = 24 * (expand_gather_us + restore_gather_us) + 2 * expand_gather_us
    total_split = 24 * (expand_split_us + restore_split_us) + 2 * expand_split_us

    return {
        'N': N,
        'prefix_len': prefix_len,
        'seq_lens': seq_lens,
        'req_lens': req_lens,
        'T_compact': T_compact,
        'T_expanded': T_expanded,
        'expand_gather': expand_gather_us,
        'expand_split': expand_split_us,
        'restore_gather': restore_gather_us,
        'restore_split': restore_split_us,
        'total_gather': total_gather,
        'total_split': total_split,
        'correct_expand': correct_expand,
        'correct_restore': correct_restore,
    }


def main():
    parser = argparse.ArgumentParser(
        description='GatherV2 vs SplitV+ConcatV2 benchmark for prefix sharing expand/restore',
    )
    parser.add_argument('--device', type=int, default=8, help='NPU device ID (default: 8)')
    parser.add_argument('--n-list', type=str, default='1,3,5,8,10',
                        help='Comma-separated N values to test (default: 1,3,5,8,10)')
    parser.add_argument('--iters', type=int, default=1000, help='Measurement iterations (default: 1000)')
    parser.add_argument('--warmup', type=int, default=50, help='Warmup iterations (default: 50)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    args = parser.parse_args()

    torch.npu.set_device(args.device)
    device = f'npu:{args.device}'

    D = 896
    dtype = torch.float16
    N_list = [int(x) for x in args.n_list.split(',')]

    print("=" * 80)
    print("GatherV2 vs SplitV+ConcatV2 Benchmark")
    print(f"  D={D}, dtype=fp16, avg_seq=150, p99=218, prefix=20-25")
    print(f"  warmup={args.warmup}, iters={args.iters}, seed={args.seed}")
    print(f"  device={device}")
    print("=" * 80)

    results = []
    for N in N_list:
        r = run_single_config(N, D, dtype, device, seed=args.seed)
        results.append(r)

        print(f"\n[N={N}]  P={r['prefix_len']}  T_c={r['T_compact']}  T_e={r['T_expanded']}")
        print(f"  seq_lens={r['seq_lens']}")
        print(f"  expand:  gather={r['expand_gather']:7.2f} us  split={r['expand_split']:7.2f} us  "
              f"ratio={r['expand_split']/r['expand_gather']:.2f}x  correct={r['correct_expand']}")
        print(f"  restore: gather={r['restore_gather']:7.2f} us  split={r['restore_split']:7.2f} us  "
              f"ratio={r['restore_split']/r['restore_gather']:.2f}x  correct={r['correct_restore']}")
        print(f"  24-layer total (50 calls): gather={r['total_gather']:8.1f} us  "
              f"split={r['total_split']:8.1f} us  diff={r['total_split']-r['total_gather']:+8.1f} us")

    print(f"\n{'=' * 80}")
    print("Summary (24-layer total, us)")
    print(f"{'N':>4}  {'gather':>10}  {'split':>10}  {'winner':>8}  {'diff':>10}")
    print(f"{'-' * 4}  {'-' * 10}  {'-' * 10}  {'-' * 8}  {'-' * 10}")
    for r in results:
        winner = "gather" if r['total_gather'] < r['total_split'] else "split"
        diff = r['total_split'] - r['total_gather']
        print(f"{r['N']:>4}  {r['total_gather']:>10.1f}  {r['total_split']:>10.1f}  "
              f"{winner:>8}  {diff:>+10.1f}")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    main()
