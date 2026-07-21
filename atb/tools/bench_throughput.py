#!/usr/bin/env python3
"""
Multi-stream throughput benchmark for Qwen2.5-0.5B OM model.

Architecture:
  Producer (1 thread) -> Queue -> Consumers (N streams)

Producer generates requests with realistic distributions:
  - batch_size: avg=9.8, p99=10.9 (normal around 10)
  - seq_len:    avg=150, p99=218 (lognormal)

Latency measured per request:
  - e2e:   enqueue -> inference complete
  - queue: enqueue -> dequeue (waiting in queue)
  - infer: ExecuteAsync start -> SyncStream complete

Usage:
  # Single run
  python bench_throughput.py --model model.om --streams 4 --requests 10000 --device-id 2

  # Sweep multiple stream counts
  python bench_throughput.py --model model.om --sweep 1,2,4,8,16 --requests 10000 --device-id 2
"""

import os
import sys
import time
import threading
import queue as queue_mod
import argparse
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import acl

# === ACL Constants (from CANN acl_base.h) ===
ACL_SUCCESS = 0
ACL_MEM_MALLOC_HUGE_FIRST = 0
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_FLOAT16 = 1
ACL_INT64 = 8
ACL_FORMAT_ND = 2

# === Model Constants ===
VOCAB_SIZE = 151936
HEAD_DIM = 64
ROPE_BASE = 1000000  # Qwen2.5 rope_theta
MAX_BATCH_SIZE = 11  # tp99=10.9
MAX_SEQ_LEN = 218    # tp99=218
MAX_TOTAL_TOKENS = MAX_BATCH_SIZE * MAX_SEQ_LEN  # 2398


# =============================================================================
# ACL Helpers
# =============================================================================

def check_ret(result, msg):
    ret = result[-1] if isinstance(result, tuple) else result
    if ret != ACL_SUCCESS:
        raise RuntimeError(f"[ACL ERROR] {msg}, ret={ret}")


def np_to_host_ptr(np_array) -> int:
    return int(np_array.ctypes.data)


# =============================================================================
# Distribution Sampling
# =============================================================================

def sample_batch_size(rng) -> int:
    """avg=9.8, p99=10.9 -> normal(9.8, 0.35) clipped [1, 11]"""
    bs = int(round(rng.normal(9.8, 0.35)))
    return max(1, min(MAX_BATCH_SIZE, bs))


def sample_seq_len(rng) -> int:
    """avg=150, p99=218 -> lognormal(mu=4.997, sigma=0.167) clipped [1, 218]"""
    sl = int(rng.lognormal(4.997, 0.167))
    return max(1, min(MAX_SEQ_LEN, sl))


# =============================================================================
# Cos/Sin Table
# =============================================================================

def precompute_cos_sin_table(max_len: int, dim: int = HEAD_DIM, base: float = ROPE_BASE):
    inv_freq = 1.0 / (base ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
    positions = np.arange(max_len, dtype=np.float32)
    freqs = np.outer(positions, inv_freq)
    emb = np.concatenate([freqs, freqs], axis=-1)
    cos_table = np.cos(emb).astype(np.float16)
    sin_table = np.sin(emb).astype(np.float16)
    return cos_table, sin_table


# =============================================================================
# Request
# =============================================================================

@dataclass
class Request:
    req_id: int
    input_ids: np.ndarray
    actual_seq_lengths: np.ndarray
    cos: np.ndarray
    sin: np.ndarray
    total_tokens: int
    batch_size: int
    enqueue_time: float = 0.0
    dequeue_time: float = 0.0
    infer_start: float = 0.0
    infer_end: float = 0.0


def generate_request(req_id: int, cos_table: np.ndarray, sin_table: np.ndarray, rng) -> Request:
    batch_size = sample_batch_size(rng)
    seq_lens = [sample_seq_len(rng) for _ in range(batch_size)]
    total_tokens = sum(seq_lens)

    input_ids = np.zeros(total_tokens, dtype=np.int64)
    position_ids = np.concatenate([np.arange(s) for s in seq_lens])
    actual_seq_lengths = np.cumsum(seq_lens).astype(np.int64)

    cos = cos_table[position_ids][np.newaxis, :, :]  # [1, T, 64]
    sin = sin_table[position_ids][np.newaxis, :, :]

    return Request(
        req_id=req_id, input_ids=input_ids,
        actual_seq_lengths=actual_seq_lengths, cos=cos, sin=sin,
        total_tokens=total_tokens, batch_size=batch_size,
    )


# =============================================================================
# Stream Context
# =============================================================================

class StreamContext:
    def __init__(self, stream_id: int, model_id: int, model_desc):
        self.stream_id = stream_id
        self.model_id = model_id
        self.model_desc = model_desc
        self.stream = None
        self.input_dataset = None
        self.output_dataset = None
        self.input_buffers: List[int] = []
        self.input_max_sizes: List[int] = []
        self.output_buffers: List[int] = []
        self.output_max_sizes: List[int] = []

        self._input_specs = [
            (ACL_INT64,   [MAX_BATCH_SIZE],                MAX_BATCH_SIZE * 8),           # actual_seq_lengths
            (ACL_FLOAT16, [1, MAX_TOTAL_TOKENS, HEAD_DIM], 1 * MAX_TOTAL_TOKENS * HEAD_DIM * 2),  # cos
            (ACL_FLOAT16, [1, MAX_TOTAL_TOKENS, HEAD_DIM], 1 * MAX_TOTAL_TOKENS * HEAD_DIM * 2),  # sin
            (ACL_INT64,   [MAX_TOTAL_TOKENS],              MAX_TOTAL_TOKENS * 8),         # input_ids
        ]
        self._max_output_size = MAX_BATCH_SIZE * VOCAB_SIZE * 2  # [N, vocab] float16

    def init(self):
        self.stream, ret = acl.rt.create_stream()
        check_ret(ret, f"create_stream {self.stream_id}")

        self.input_dataset = acl.mdl.create_dataset()
        for dtype, shape, max_size in self._input_specs:
            ptr, ret = acl.rt.malloc(max_size, ACL_MEM_MALLOC_HUGE_FIRST)
            check_ret(ret, f"malloc input stream {self.stream_id}")
            buf = acl.create_data_buffer(ptr, max_size)
            ret = acl.mdl.add_dataset_buffer(self.input_dataset, buf)
            check_ret(ret, f"add_dataset_buffer input")
            self.input_buffers.append(ptr)
            self.input_max_sizes.append(max_size)

        self.output_dataset = acl.mdl.create_dataset()
        ptr, ret = acl.rt.malloc(self._max_output_size, ACL_MEM_MALLOC_HUGE_FIRST)
        check_ret(ret, f"malloc output stream {self.stream_id}")
        buf = acl.create_data_buffer(ptr, self._max_output_size)
        ret = acl.mdl.add_dataset_buffer(self.output_dataset, buf)
        check_ret(ret, f"add_dataset_buffer output")
        self.output_buffers.append(ptr)
        self.output_max_sizes.append(self._max_output_size)

    def set_inputs(self, req: Request):
        inputs = [
            (req.actual_seq_lengths, ACL_INT64,   list(req.actual_seq_lengths.shape)),
            (req.cos,                ACL_FLOAT16, list(req.cos.shape)),
            (req.sin,                ACL_FLOAT16, list(req.sin.shape)),
            (req.input_ids,          ACL_INT64,   list(req.input_ids.shape)),
        ]
        for i, (data, dtype, shape) in enumerate(inputs):
            data_size = data.nbytes
            if data_size > self.input_max_sizes[i]:
                raise RuntimeError(f"Data {data_size} > buffer {self.input_max_sizes[i]}")
            ret = acl.rt.memcpy(
                self.input_buffers[i], self.input_max_sizes[i],
                np_to_host_ptr(data), data_size, ACL_MEMCPY_HOST_TO_DEVICE,
            )
            check_ret(ret, f"H2D input[{i}]")
            desc = acl.create_tensor_desc(dtype, shape, ACL_FORMAT_ND)
            ret = acl.mdl.set_dataset_tensor_desc(self.input_dataset, desc, i)
            check_ret(ret, f"set_desc input[{i}]")
            acl.destroy_tensor_desc(desc)

    def execute(self):
        ret = acl.mdl.execute_async(
            self.model_id, self.input_dataset, self.output_dataset, self.stream,
        )
        check_ret(ret, f"execute_async stream {self.stream_id}")
        ret = acl.rt.synchronize_stream(self.stream)
        check_ret(ret, f"sync_stream {self.stream_id}")

    def cleanup(self):
        for ds, buffers in [
            (self.input_dataset, self.input_buffers),
            (self.output_dataset, self.output_buffers),
        ]:
            if ds is not None:
                for i in range(acl.mdl.get_dataset_num_buffers(ds)):
                    buf = acl.mdl.get_dataset_buffer(ds, i)
                    if buf:
                        ptr = acl.get_data_buffer_addr(buf)
                        if ptr:
                            acl.rt.free(ptr)
                        acl.destroy_data_buffer(buf)
                acl.mdl.destroy_dataset(ds)
        if self.stream:
            acl.rt.destroy_stream(self.stream)


# =============================================================================
# Stats
# =============================================================================

class Stats:
    def __init__(self):
        self.lock = threading.Lock()
        self.e2e_times: List[float] = []
        self.queue_times: List[float] = []
        self.infer_times: List[float] = []
        self.total_tokens = 0
        self.per_stream: Dict[int, List[float]] = defaultdict(list)

    def record(self, req: Request, stream_id: int):
        e2e = (req.infer_end - req.enqueue_time) * 1000
        qw = (req.dequeue_time - req.enqueue_time) * 1000
        inf = (req.infer_end - req.infer_start) * 1000
        with self.lock:
            self.e2e_times.append(e2e)
            self.queue_times.append(qw)
            self.infer_times.append(inf)
            self.total_tokens += req.total_tokens
            self.per_stream[stream_id].append(inf)

    @staticmethod
    def _pct(data, p):
        if not data:
            return 0.0
        s = sorted(data)
        return s[min(int(len(s) * p / 100), len(s) - 1)]

    def report(self, total_ms: float, num_streams: int, warmup: int):
        n = len(self.e2e_times)
        if n == 0:
            print("[WARN] No requests completed")
            return

        qps = n / (total_ms / 1000)
        tps = self.total_tokens / (total_ms / 1000)

        def line(name, data):
            avg = np.mean(data)
            print(f"  {name:20s} avg={avg:8.2f}  p50={self._pct(data,50):8.2f}  "
                  f"p90={self._pct(data,90):8.2f}  p99={self._pct(data,99):8.2f}  "
                  f"max={max(data):8.2f}  (ms)")

        print(f"\n{'=' * 72}")
        print(f"Throughput Benchmark Results")
        print(f"{'=' * 72}")
        print(f"  Streams:          {num_streams}")
        print(f"  Requests:         {n} (warmup={warmup})")
        print(f"  Total Time:       {total_ms:.2f} ms")
        print(f"  Total Tokens:     {self.total_tokens:,}")
        print(f"  {'-' * 68}")
        print(f"  QPS:              {qps:.2f} req/s")
        print(f"  Token Throughput: {tps:,.0f} tokens/s")
        print(f"  {'-' * 68}")
        print(f"  Latency (ms):")
        line("E2E", self.e2e_times)
        line("Queue Wait", self.queue_times)
        line("Inference", self.infer_times)
        print(f"  {'-' * 68}")
        print(f"  Per-Stream:")
        for sid in sorted(self.per_stream.keys()):
            times = self.per_stream[sid]
            print(f"    Stream {sid}: {len(times)} reqs, avg_infer={np.mean(times):.2f}ms")
        print(f"{'=' * 72}\n")

    def summary(self) -> dict:
        return {
            'qps': len(self.e2e_times) / max(1e-9, 1),
            'e2e_avg': float(np.mean(self.e2e_times)) if self.e2e_times else 0,
            'e2e_p99': self._pct(self.e2e_times, 99),
            'infer_avg': float(np.mean(self.infer_times)) if self.infer_times else 0,
            'infer_p99': self._pct(self.infer_times, 99),
            'total_tokens': self.total_tokens,
        }


# =============================================================================
# Threads
# =============================================================================

def producer_thread(task_queue, total, cos_table, sin_table, seed=42):
    rng = np.random.default_rng(seed)
    for i in range(total):
        req = generate_request(i, cos_table, sin_table, rng)
        req.enqueue_time = time.perf_counter()
        task_queue.put(req)


def consumer_thread(ctx: StreamContext, task_queue, stats: Stats, active: threading.Event,
                    acl_context=None):
    if acl_context is not None:
        acl.rt.set_context(acl_context)
    active.set()
    while True:
        try:
            req = task_queue.get(timeout=0.5)
        except queue_mod.Empty:
            if not active.is_set():
                break
            continue
        if req is None:
            break
        try:
            req.dequeue_time = time.perf_counter()
            ctx.set_inputs(req)
            req.infer_start = time.perf_counter()
            ctx.execute()
            req.infer_end = time.perf_counter()
            stats.record(req, ctx.stream_id)
        except Exception as e:
            print(f"[ERROR] Stream {ctx.stream_id} req {req.req_id}: {e}", file=sys.stderr)
            break


# =============================================================================
# Benchmark Runner
# =============================================================================

def run_benchmark(model_id, model_desc, num_streams, total_requests,
                  cos_table, sin_table, device_id, warmup, queue_depth):
    print(f"\n{'=' * 72}")
    print(f"Config: streams={num_streams}, requests={total_requests}, "
          f"warmup={warmup}, device={device_id}")
    print(f"{'=' * 72}")

    acl_context, ret = acl.rt.get_context()
    check_ret(ret, "get_context")

    streams = []
    for sid in range(num_streams):
        ctx = StreamContext(sid, model_id, model_desc)
        ctx.init()
        streams.append(ctx)
    print(f"[INFO] Created {num_streams} stream contexts")

    if warmup > 0:
        print(f"[INFO] Warmup ({warmup} requests)...")
        rng = np.random.default_rng(0)
        for i in range(warmup):
            req = generate_request(i, cos_table, sin_table, rng)
            streams[i % num_streams].set_inputs(req)
            streams[i % num_streams].execute()
        print(f"[INFO] Warmup done")

    task_queue = queue_mod.Queue(maxsize=queue_depth)
    stats = Stats()
    active = threading.Event()

    consumers = []
    for ctx in streams:
        t = threading.Thread(target=consumer_thread,
                             args=(ctx, task_queue, stats, active, acl_context))
        t.daemon = True
        t.start()
        consumers.append(t)

    for t in consumers:
        active.set()

    bench_start = time.perf_counter()
    prod = threading.Thread(target=producer_thread,
                            args=(task_queue, total_requests, cos_table, sin_table))
    prod.daemon = True
    prod.start()
    prod.join(timeout=300)

    retry = 0
    while not task_queue.empty() and retry < 600:
        time.sleep(0.5)
        retry += 1

    active.clear()
    for _ in consumers:
        task_queue.put(None)
    for t in consumers:
        t.join(timeout=10)
    bench_end = time.perf_counter()

    total_ms = (bench_end - bench_start) * 1000
    stats.report(total_ms, num_streams, warmup)

    for ctx in streams:
        ctx.cleanup()

    return stats, total_ms


def print_sweep_table(results: List[Tuple[int, Stats, float]]):
    print(f"\n{'=' * 90}")
    print(f"Sweep Results: QPS vs Stream Count")
    print(f"{'=' * 90}")
    header = (f"{'Streams':>7s} | {'QPS':>10s} | {'Token/s':>12s} | "
              f"{'E2E avg':>10s} | {'E2E p99':>10s} | "
              f"{'Infer avg':>10s} | {'Infer p99':>10s}")
    print(header)
    print(f"{'-' * 90}")
    for n, stats, total_ms in results:
        qps = len(stats.e2e_times) / (total_ms / 1000)
        tps = stats.total_tokens / (total_ms / 1000)
        e2e_avg = np.mean(stats.e2e_times) if stats.e2e_times else 0
        e2e_p99 = Stats._pct(stats.e2e_times, 99)
        inf_avg = np.mean(stats.infer_times) if stats.infer_times else 0
        inf_p99 = Stats._pct(stats.infer_times, 99)
        print(f"{n:>7d} | {qps:>10.2f} | {tps:>12,.0f} | "
              f"{e2e_avg:>10.2f} | {e2e_p99:>10.2f} | "
              f"{inf_avg:>10.2f} | {inf_p99:>10.2f}")
    print(f"{'=' * 90}\n")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Multi-stream throughput benchmark for Qwen2.5-0.5B OM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--model', required=True, help='Path to .om model')
    parser.add_argument('--streams', type=int, default=4, help='Number of streams (default: 4)')
    parser.add_argument('--sweep', type=str, default=None,
                        help='Comma-separated stream counts to sweep (e.g. 1,2,4,8)')
    parser.add_argument('--requests', type=int, default=10000, help='Total requests (default: 10000)')
    parser.add_argument('--device-id', type=int, default=0, help='NPU device ID (default: 0)')
    parser.add_argument('--warmup', type=int, default=50, help='Warmup requests (default: 50)')
    parser.add_argument('--queue-depth', type=int, default=None,
                        help='Queue depth (default: streams * 4)')
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"[ERROR] Model not found: {args.model}")
        sys.exit(1)

    if args.sweep:
        stream_counts = [int(x) for x in args.sweep.split(',')]
    else:
        stream_counts = [args.streams]

    ret = acl.init()
    check_ret(ret, "acl.init")
    ret = acl.rt.set_device(args.device_id)
    check_ret(ret, f"set_device({args.device_id})")

    model_id, ret = acl.mdl.load_from_file(args.model)
    check_ret(ret, f"load_from_file")
    model_desc = acl.mdl.create_desc()
    ret = acl.mdl.get_desc(model_desc, model_id)
    check_ret(ret, "get_desc")
    ni = acl.mdl.get_num_inputs(model_desc)
    no = acl.mdl.get_num_outputs(model_desc)
    print(f"[INFO] Model loaded: inputs={ni}, outputs={no}")

    cos_table, sin_table = precompute_cos_sin_table(MAX_SEQ_LEN)
    print(f"[INFO] Cos/sin table: [{MAX_SEQ_LEN}, {HEAD_DIM}]")

    results = []
    for n in stream_counts:
        qd = args.queue_depth or (n * 4)
        stats, total_ms = run_benchmark(
            model_id, model_desc, n, args.requests,
            cos_table, sin_table, args.device_id, args.warmup, qd,
        )
        results.append((n, stats, total_ms))

    if len(results) > 1:
        print_sweep_table(results)

    acl.mdl.unload(model_id)
    acl.mdl.destroy_desc(model_desc)
    acl.rt.reset_device(args.device_id)
    acl.finalize()


if __name__ == '__main__':
    main()
