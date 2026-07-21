#!/usr/bin/env python3
"""
Parse CANN profiling data using the official msprof tool and exported CSV/JSON files.

Profiling workflow (CANN official):
  1. Collect: acl.json with profiler config -> PROF_XXX directories
  2. Parse:   msprof --parse=on --output=<PROF_XXX>
  3. Export:  msprof --export=on --output=<PROF_XXX> -> mindstudio_profiler_output/
  4. Query:  msprof --query=on --output=<PROF_XXX>

This script wraps msprof and provides analysis on exported CSV/JSON files:
  - op_summary:    Per-operator detailed metrics (task time, AI Core, memory)
  - op_statistic:  Aggregated statistics by operator type
  - task_time:     Kernel/task execution timeline
  - api_statistic: Host API call statistics
  - step_trace:    Iteration/step timing
  - fusion_op:     Fusion operator memory usage

Usage:
    python parse_profiling.py parse  --profiling_dir ./profiling_data
    python parse_profiling.py export --profiling_dir ./profiling_data
    python parse_profiling.py query  --profiling_dir ./profiling_data
    python parse_profiling.py summary --profiling_dir ./profiling_data
    python parse_profiling.py show --profiling_dir ./profiling_data --op "Softmax"
    python parse_profiling.py top --profiling_dir ./profiling_data --sort time --top_n 20
    python parse_profiling.py timeline --profiling_dir ./profiling_data
    python parse_profiling.py api --profiling_dir ./profiling_data
"""

import os
import sys
import argparse
import glob
import csv
import json
import subprocess
from pathlib import Path
from collections import defaultdict

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def find_msprof():
    """Auto-detect msprof tool location."""
    search_paths = [
        os.environ.get('ASCEND_HOME', ''),
        '/usr/local/Ascend/ascend-toolkit/latest',
        '/usr/local/Ascend/cann-8.5.0',
        '/usr/local/Ascend/latest',
    ]
    for base in search_paths:
        if not base:
            continue
        candidate = os.path.join(base, 'bin', 'msprof')
        if os.path.isfile(candidate):
            return candidate
    path_result = subprocess.run(['which', 'msprof'], capture_output=True, text=True)
    if path_result.returncode == 0:
        return path_result.stdout.strip()
    return None


def find_prof_dirs(profiling_dir):
    """Find all PROF_XXX directories under profiling_dir."""
    dirs = []
    for item in os.listdir(profiling_dir):
        if item.startswith('PROF_'):
            full_path = os.path.join(profiling_dir, item)
            if os.path.isdir(full_path):
                dirs.append(full_path)
    return sorted(dirs)


def find_exported_csv(prof_dir):
    """Find exported CSV files in mindstudio_profiler_output/."""
    output_dir = os.path.join(prof_dir, 'mindstudio_profiler_output')
    if not os.path.isdir(output_dir):
        return {}
    result = {}
    for f in os.listdir(output_dir):
        if f.endswith('.csv'):
            if 'op_summary' in f:
                result['op_summary'] = os.path.join(output_dir, f)
            elif 'op_statistic' in f:
                result['op_statistic'] = os.path.join(output_dir, f)
            elif 'task_time' in f:
                result['task_time'] = os.path.join(output_dir, f)
            elif 'api_statistic' in f:
                result['api_statistic'] = os.path.join(output_dir, f)
            elif 'step_trace' in f:
                result['step_trace'] = os.path.join(output_dir, f)
            elif 'fusion_op' in f:
                result['fusion_op'] = os.path.join(output_dir, f)
    return result


def read_csv(filepath):
    """Read CSV file and return list of dicts."""
    rows = []
    try:
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    except Exception as e:
        print(f"[ERROR] Failed to read {filepath}: {e}")
    return rows


def safe_float(val, default=0.0):
    """Safely convert a value to float."""
    try:
        v = val.strip().replace('\t', '') if isinstance(val, str) else val
        return float(v)
    except (ValueError, TypeError, AttributeError):
        return default


# =============================================================================
# msprof wrapper commands
# =============================================================================

def run_msprof_parse(profiling_dir, msprof_path=None):
    """Run msprof --parse on PROF_XXX directories."""
    if msprof_path is None:
        msprof_path = find_msprof()
    if not msprof_path:
        print("[ERROR] msprof not found. Set ASCEND_HOME or add to PATH.")
        return False

    prof_dirs = find_prof_dirs(profiling_dir)
    if not prof_dirs:
        print(f"[INFO] No PROF_XXX directories found in {profiling_dir}")
        return False

    for prof_dir in prof_dirs:
        print(f"[INFO] Parsing: {os.path.basename(prof_dir)}")
        cmd = [msprof_path, '--parse=on', f'--output={prof_dir}']
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode != 0:
                print(f"[ERROR] Parse failed: {result.stderr[-500:] if result.stderr else 'unknown error'}")
            else:
                print(f"[INFO] Parse complete: {os.path.basename(prof_dir)}")
        except subprocess.TimeoutExpired:
            print(f"[ERROR] Parse timed out for {os.path.basename(prof_dir)}")
        except Exception as e:
            print(f"[ERROR] Parse failed: {e}")
    return True


def run_msprof_export(profiling_dir, msprof_path=None):
    """Run msprof --export on PROF_XXX directories."""
    if msprof_path is None:
        msprof_path = find_msprof()
    if not msprof_path:
        print("[ERROR] msprof not found. Set ASCEND_HOME or add to PATH.")
        return False

    prof_dirs = find_prof_dirs(profiling_dir)
    if not prof_dirs:
        print(f"[INFO] No PROF_XXX directories found in {profiling_dir}")
        return False

    for prof_dir in prof_dirs:
        print(f"[INFO] Exporting: {os.path.basename(prof_dir)}")
        cmd = [msprof_path, '--export=on', f'--output={prof_dir}']
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode != 0:
                print(f"[ERROR] Export failed: {result.stderr[-500:] if result.stderr else 'unknown error'}")
            else:
                print(f"[INFO] Export complete: {os.path.basename(prof_dir)}")
        except subprocess.TimeoutExpired:
            print(f"[ERROR] Export timed out for {os.path.basename(prof_dir)}")
        except Exception as e:
            print(f"[ERROR] Export failed: {e}")
    return True


def run_msprof_query(profiling_dir, msprof_path=None):
    """Run msprof --query on PROF_XXX directories."""
    if msprof_path is None:
        msprof_path = find_msprof()
    if not msprof_path:
        print("[ERROR] msprof not found. Set ASCEND_HOME or add to PATH.")
        return False

    prof_dirs = find_prof_dirs(profiling_dir)
    if not prof_dirs:
        print(f"[INFO] No PROF_XXX directories found in {profiling_dir}")
        return False

    for prof_dir in prof_dirs:
        print(f"[INFO] Querying: {os.path.basename(prof_dir)}")
        cmd = [msprof_path, '--query=on', f'--output={prof_dir}']
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.stdout:
                print(result.stdout)
            if result.returncode != 0 and result.stderr:
                print(f"[ERROR] {result.stderr[-500:]}")
        except Exception as e:
            print(f"[ERROR] Query failed: {e}")
    return True


# =============================================================================
# Analysis commands
# =============================================================================

def cmd_parse_and_export(profiling_dir, msprof_path=None):
    """Run parse + export in one step."""
    print("=" * 80)
    print("Step 1: Parse profiling data")
    print("=" * 80)
    run_msprof_parse(profiling_dir, msprof_path)
    print()
    print("=" * 80)
    print("Step 2: Export profiling data")
    print("=" * 80)
    run_msprof_export(profiling_dir, msprof_path)


def cmd_list(profiling_dir):
    """List all PROF_XXX sessions and their exported files."""
    prof_dirs = find_prof_dirs(profiling_dir)
    if not prof_dirs:
        print(f"[INFO] No PROF_XXX directories found in {profiling_dir}")
        return

    print(f"\n{'=' * 80}")
    print(f"Profiling Sessions in {profiling_dir}")
    print(f"{'=' * 80}")
    print(f"Total sessions: {len(prof_dirs)}")
    print()

    for i, prof_dir in enumerate(prof_dirs, 1):
        name = os.path.basename(prof_dir)
        csv_files = find_exported_csv(prof_dir)
        has_device = any(d.startswith('device_') for d in os.listdir(prof_dir) if os.path.isdir(os.path.join(prof_dir, d)))
        has_host = os.path.isdir(os.path.join(prof_dir, 'host'))
        has_sqlite = any(
            os.path.isdir(os.path.join(prof_dir, d, 'sqlite'))
            for d in os.listdir(prof_dir) if os.path.isdir(os.path.join(prof_dir, d))
        )
        has_output = os.path.isdir(os.path.join(prof_dir, 'mindstudio_profiler_output'))

        print(f"{i}. {name}")
        print(f"   Parsed:  {'Yes' if has_sqlite else 'No (run: parse_profiling.py parse)'}")
        print(f"   Exported: {'Yes' if has_output and csv_files else 'No (run: parse_profiling.py export)'}")
        print(f"   Device:  {'Yes' if has_device else 'No'}")
        print(f"   Host:    {'Yes' if has_host else 'No'}")
        if csv_files:
            print(f"   CSV files:")
            for csv_type, csv_path in sorted(csv_files.items()):
                size_kb = os.path.getsize(csv_path) / 1024
                print(f"     - {csv_type:20s} {size_kb:8.1f} KB")
        print()


def cmd_summary(profiling_dir):
    """Show profiling summary from exported CSV files."""
    prof_dirs = find_prof_dirs(profiling_dir)
    if not prof_dirs:
        print(f"[INFO] No PROF_XXX directories found in {profiling_dir}")
        return

    prof_dir = prof_dirs[-1]
    csv_files = find_exported_csv(prof_dir)

    if not csv_files:
        print(f"[INFO] No exported CSV files found in {prof_dir}")
        print("       Run 'parse_profiling.py parse' and 'parse_profiling.py export' first")
        return

    print(f"\n{'=' * 80}")
    print(f"Profiling Summary: {os.path.basename(prof_dir)}")
    print(f"{'=' * 80}")

    # op_statistic: aggregated by operator type
    if 'op_statistic' in csv_files:
        rows = read_csv(csv_files['op_statistic'])
        if rows:
            print(f"\nOperator Statistics (by type):")
            print(f"{'OP Type':40s} {'Core Type':15s} {'Count':>6s} {'Total(us)':>12s} {'Avg(us)':>10s} {'Max(us)':>10s} {'Ratio(%)':>10s}")
            print("-" * 105)
            for row in sorted(rows, key=lambda r: safe_float(r.get('Total Time(us)', 0)), reverse=True):
                print(f"{row.get('OP Type', 'N/A'):40s} "
                      f"{row.get('Core Type', 'N/A'):15s} "
                      f"{row.get('Count', 'N/A'):>6s} "
                      f"{safe_float(row.get('Total Time(us)', 0)):>12.2f} "
                      f"{safe_float(row.get('Avg Time(us)', 0)):>10.2f} "
                      f"{safe_float(row.get('Max Time(us)', 0)):>10.2f} "
                      f"{safe_float(row.get('Ratio(%)', 0)):>10.2f}")
            print()

    # step_trace
    if 'step_trace' in csv_files:
        rows = read_csv(csv_files['step_trace'])
        if rows:
            print(f"Step Trace:")
            for row in rows:
                iter_time = safe_float(row.get('Iteration Time(us)', 0))
                model_id = row.get('Model ID', 'N/A')
                iter_id = row.get('Iteration ID', 'N/A')
                print(f"  Model ID: {model_id}, Iteration: {iter_id}, "
                      f"Time: {iter_time:.2f} us ({iter_time/1000:.2f} ms)")
            print()

    # api_statistic: top 10 API calls
    if 'api_statistic' in csv_files:
        rows = read_csv(csv_files['api_statistic'])
        if rows:
            print(f"Top 10 API Calls (by total time):")
            print(f"{'Level':10s} {'API Name':40s} {'Time(us)':>12s} {'Count':>8s} {'Avg(us)':>10s}")
            print("-" * 82)
            for row in sorted(rows, key=lambda r: safe_float(r.get('Time(us)', 0)), reverse=True)[:10]:
                print(f"{row.get('Level', 'N/A'):10s} "
                      f"{row.get('API Name', 'N/A'):40s} "
                      f"{safe_float(row.get('Time(us)', 0)):>12.2f} "
                      f"{row.get('Count', 'N/A'):>8s} "
                      f"{safe_float(row.get('Avg(us)', 0)):>10.2f}")
            print()


def cmd_show_op(profiling_dir, op_name):
    """Show detailed info for a specific operator from op_summary."""
    prof_dirs = find_prof_dirs(profiling_dir)
    if not prof_dirs:
        print(f"[INFO] No PROF_XXX directories found in {profiling_dir}")
        return

    prof_dir = prof_dirs[-1]
    csv_files = find_exported_csv(prof_dir)

    if 'op_summary' not in csv_files:
        print("[INFO] op_summary CSV not found. Run export first.")
        return

    rows = read_csv(csv_files['op_summary'])
    matching = [r for r in rows if op_name.lower() in r.get('Op Name', '').lower()]

    if not matching:
        print(f"[INFO] No operators matching '{op_name}' found")
        return

    print(f"\n{'=' * 80}")
    print(f"Operator Details: {op_name} ({len(matching)} matches)")
    print(f"{'=' * 80}")

    for i, row in enumerate(matching[:20], 1):
        print(f"\n--- Match {i}: {row.get('Op Name', 'N/A')} ---")
        print(f"  OP Type:          {row.get('OP Type', 'N/A')}")
        print(f"  Task Type:        {row.get('Task Type', 'N/A')}")
        print(f"  OP State:         {row.get('OP State', 'N/A')}")
        print(f"  Task ID:          {row.get('Task ID', 'N/A')}")
        print(f"  Stream ID:        {row.get('Stream ID', 'N/A')}")
        print(f"  Task Duration:    {safe_float(row.get('Task Duration(us)', 0)):.3f} us")
        print(f"  Task Wait Time:   {safe_float(row.get('Task Wait Time(us)', 0)):.3f} us")
        print(f"  Input Shapes:     {row.get('Input Shapes', 'N/A')}")
        print(f"  Input Data Types: {row.get('Input Data Types', 'N/A')}")
        print(f"  Output Shapes:    {row.get('Output Shapes', 'N/A')}")

        aicore_time = safe_float(row.get('aicore_time(us)', 0))
        aiv_time = safe_float(row.get('aiv_time(us)', 0))
        if aicore_time > 0 or aiv_time > 0:
            print(f"  AI Core Time:     {aicore_time:.3f} us")
            print(f"    mac ratio:      {row.get('aic_mac_ratio', 'N/A')}")
            print(f"    scalar ratio:   {row.get('aic_scalar_ratio', 'N/A')}")
            print(f"    mte1 ratio:     {row.get('aic_mte1_ratio', 'N/A')}")
            print(f"    mte2 ratio:     {row.get('aic_mte2_ratio', 'N/A')}")
            print(f"    fixpipe ratio:  {row.get('aic_fixpipe_ratio', 'N/A')}")
            print(f"  AI Vector Time:   {aiv_time:.3f} us")

    if len(matching) > 20:
        print(f"\n... and {len(matching) - 20} more matches")


def cmd_top(profiling_dir, sort_by='time', top_n=20):
    """Show top N operators by time or count."""
    prof_dirs = find_prof_dirs(profiling_dir)
    if not prof_dirs:
        print(f"[INFO] No PROF_XXX directories found in {profiling_dir}")
        return

    prof_dir = prof_dirs[-1]
    csv_files = find_exported_csv(prof_dir)

    if 'op_summary' not in csv_files:
        print("[INFO] op_summary CSV not found. Run export first.")
        return

    rows = read_csv(csv_files['op_summary'])
    if not rows:
        print("[INFO] No operator data found")
        return

    if sort_by == 'time':
        sorted_rows = sorted(rows, key=lambda r: safe_float(r.get('Task Duration(us)', 0)), reverse=True)
        sort_label = 'Task Duration(us)'
    elif sort_by == 'count':
        op_counts = defaultdict(int)
        for r in rows:
            op_counts[r.get('OP Type', 'Unknown')] += 1
        print(f"\n{'=' * 80}")
        print(f"Top {top_n} Operator Types (by count)")
        print(f"{'=' * 80}")
        print(f"{'OP Type':40s} {'Count':>8s}")
        print("-" * 50)
        for op_type, count in sorted(op_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]:
            print(f"{op_type:40s} {count:>8d}")
        return
    else:
        print(f"[ERROR] Unknown sort key: {sort_by}. Use 'time' or 'count'.")
        return

    print(f"\n{'=' * 80}")
    print(f"Top {top_n} Operators (by {sort_label})")
    print(f"{'=' * 80}")
    print(f"{'Op Name':50s} {'OP Type':20s} {'Duration(us)':>14s} {'AI Core(us)':>12s}")
    print("-" * 98)
    for row in sorted_rows[:top_n]:
        name = row.get('Op Name', 'N/A')
        if len(name) > 48:
            name = '...' + name[-45:]
        print(f"{name:50s} "
              f"{row.get('OP Type', 'N/A'):20s} "
              f"{safe_float(row.get('Task Duration(us)', 0)):>14.3f} "
              f"{safe_float(row.get('aicore_time(us)', 0)):>12.3f}")


def cmd_timeline(profiling_dir):
    """Show task timeline from task_time CSV."""
    prof_dirs = find_prof_dirs(profiling_dir)
    if not prof_dirs:
        print(f"[INFO] No PROF_XXX directories found in {profiling_dir}")
        return

    prof_dir = prof_dirs[-1]
    csv_files = find_exported_csv(prof_dir)

    if 'task_time' not in csv_files:
        print("[INFO] task_time CSV not found. Run export first.")
        return

    rows = read_csv(csv_files['task_time'])
    if not rows:
        print("[INFO] No task_time data found")
        return

    print(f"\n{'=' * 80}")
    print(f"Task Timeline ({len(rows)} tasks)")
    print(f"{'=' * 80}")

    # Group by kernel_type
    type_counts = defaultdict(int)
    type_times = defaultdict(float)
    for row in rows:
        kt = row.get('kernel_type', 'Unknown')
        type_counts[kt] += 1
        type_times[kt] += safe_float(row.get('task_time(us)', 0))

    print(f"\nBy kernel type:")
    print(f"{'Kernel Type':30s} {'Count':>8s} {'Total Time(us)':>16s}")
    print("-" * 56)
    for kt in sorted(type_times.keys(), key=lambda k: type_times[k], reverse=True):
        print(f"{kt:30s} {type_counts[kt]:>8d} {type_times[kt]:>16.3f}")

    # Show first 20 tasks
    print(f"\nFirst 20 tasks:")
    print(f"{'kernel_type':25s} {'stream':>6s} {'task_id':>8s} {'time(us)':>12s} {'start(us)':>24s}")
    print("-" * 77)
    for row in rows[:20]:
        print(f"{row.get('kernel_type', 'N/A'):25s} "
              f"{row.get('stream_id', 'N/A'):>6s} "
              f"{row.get('task_id', 'N/A'):>8s} "
              f"{safe_float(row.get('task_time(us)', 0)):>12.3f} "
              f"{row.get('task_start(us)', 'N/A'):>24s}")


def cmd_api(profiling_dir):
    """Show API statistics."""
    prof_dirs = find_prof_dirs(profiling_dir)
    if not prof_dirs:
        print(f"[INFO] No PROF_XXX directories found in {profiling_dir}")
        return

    prof_dir = prof_dirs[-1]
    csv_files = find_exported_csv(prof_dir)

    if 'api_statistic' not in csv_files:
        print("[INFO] api_statistic CSV not found. Run export first.")
        return

    rows = read_csv(csv_files['api_statistic'])
    if not rows:
        print("[INFO] No API data found")
        return

    print(f"\n{'=' * 80}")
    print(f"API Statistics ({len(rows)} APIs)")
    print(f"{'=' * 80}")
    print(f"{'Level':10s} {'API Name':45s} {'Time(us)':>12s} {'Count':>8s} {'Avg(us)':>10s} {'Max(us)':>10s}")
    print("-" * 97)
    for row in sorted(rows, key=lambda r: safe_float(r.get('Time(us)', 0)), reverse=True):
        print(f"{row.get('Level', 'N/A'):10s} "
              f"{row.get('API Name', 'N/A'):45s} "
              f"{safe_float(row.get('Time(us)', 0)):>12.2f} "
              f"{row.get('Count', 'N/A'):>8s} "
              f"{safe_float(row.get('Avg(us)', 0)):>10.2f} "
              f"{safe_float(row.get('Max(us)', 0)):>10.2f}")


def cmd_plot(profiling_dir, output_file=None):
    """Plot top operators by execution time."""
    if not HAS_MATPLOTLIB:
        print("[ERROR] matplotlib required for plotting. Install with: pip install matplotlib")
        return
    if not HAS_NUMPY:
        print("[ERROR] numpy required for plotting. Install with: pip install numpy")
        return

    prof_dirs = find_prof_dirs(profiling_dir)
    if not prof_dirs:
        print(f"[INFO] No PROF_XXX directories found in {profiling_dir}")
        return

    prof_dir = prof_dirs[-1]
    csv_files = find_exported_csv(prof_dir)

    if 'op_summary' not in csv_files:
        print("[INFO] op_summary CSV not found. Run export first.")
        return

    rows = read_csv(csv_files['op_summary'])
    if not rows:
        print("[INFO] No operator data found")
        return

    sorted_rows = sorted(rows, key=lambda r: safe_float(r.get('Task Duration(us)', 0)), reverse=True)
    top_n = min(30, len(sorted_rows))

    names = []
    times = []
    for row in sorted_rows[:top_n]:
        name = row.get('Op Name', 'N/A')
        if len(name) > 50:
            name = '...' + name[-47:]
        names.append(name)
        times.append(safe_float(row.get('Task Duration(us)', 0)))

    fig, ax = plt.subplots(figsize=(14, max(8, top_n * 0.4)))
    y_pos = np.arange(len(names))
    ax.barh(y_pos, times, align='center', alpha=0.7, color='steelblue')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel('Task Duration (us)')
    ax.set_title(f'Top {top_n} Operators by Execution Time\n{os.path.basename(prof_dir)}')
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"[INFO] Plot saved to: {output_file}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Parse CANN profiling data (using msprof + CSV analysis)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Workflow:
  Step 1: Collect profiling data (run inference with --profiling)
  Step 2: Parse + Export:  python parse_profiling.py parse-and-export --profiling_dir ./profiling_data
  Step 3: Analyze:        python parse_profiling.py summary --profiling_dir ./profiling_data

Commands:
  parse           Run msprof --parse (parse raw data to sqlite)
  export          Run msprof --export (export sqlite to CSV/JSON)
  parse-and-export  Run parse + export in one step
  query           Run msprof --query (query session info)
  list            List profiling sessions and exported files
  summary         Show profiling summary (op_statistic, step_trace, api_statistic)
  show            Show detailed operator info from op_summary
  top             Show top N operators by time or count
  timeline        Show task timeline from task_time
  api             Show API call statistics
  plot            Plot top operators by execution time

Examples:
  # Full workflow: parse + export + summary
  python parse_profiling.py parse-and-export --profiling_dir ./profiling_data
  python parse_profiling.py summary --profiling_dir ./profiling_data

  # Show top 20 operators by time
  python parse_profiling.py top --profiling_dir ./profiling_data --sort time --top_n 20

  # Show specific operator details
  python parse_profiling.py show --profiling_dir ./profiling_data --op "FlashAttention"

  # Plot execution time
  python parse_profiling.py plot --profiling_dir ./profiling_data --output profiling.png
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # parse
    p = subparsers.add_parser('parse', help='Run msprof --parse')
    p.add_argument('--profiling_dir', type=str, default='./profiling_data')
    p.add_argument('--msprof', type=str, default=None, help='Path to msprof binary')

    # export
    p = subparsers.add_parser('export', help='Run msprof --export')
    p.add_argument('--profiling_dir', type=str, default='./profiling_data')
    p.add_argument('--msprof', type=str, default=None)

    # parse-and-export
    p = subparsers.add_parser('parse-and-export', help='Run parse + export in one step')
    p.add_argument('--profiling_dir', type=str, default='./profiling_data')
    p.add_argument('--msprof', type=str, default=None)

    # query
    p = subparsers.add_parser('query', help='Run msprof --query')
    p.add_argument('--profiling_dir', type=str, default='./profiling_data')
    p.add_argument('--msprof', type=str, default=None)

    # list
    p = subparsers.add_parser('list', help='List profiling sessions')
    p.add_argument('--profiling_dir', type=str, default='./profiling_data')

    # summary
    p = subparsers.add_parser('summary', help='Show profiling summary')
    p.add_argument('--profiling_dir', type=str, default='./profiling_data')

    # show
    p = subparsers.add_parser('show', help='Show operator details')
    p.add_argument('--profiling_dir', type=str, default='./profiling_data')
    p.add_argument('--op', type=str, required=True, help='Operator name (partial match)')

    # top
    p = subparsers.add_parser('top', help='Show top N operators')
    p.add_argument('--profiling_dir', type=str, default='./profiling_data')
    p.add_argument('--sort', type=str, default='time', choices=['time', 'count'])
    p.add_argument('--top_n', type=int, default=20)

    # timeline
    p = subparsers.add_parser('timeline', help='Show task timeline')
    p.add_argument('--profiling_dir', type=str, default='./profiling_data')

    # api
    p = subparsers.add_parser('api', help='Show API statistics')
    p.add_argument('--profiling_dir', type=str, default='./profiling_data')

    # plot
    p = subparsers.add_parser('plot', help='Plot top operators')
    p.add_argument('--profiling_dir', type=str, default='./profiling_data')
    p.add_argument('--output', type=str, default=None, help='Save plot to file')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    if not os.path.exists(args.profiling_dir):
        print(f"[ERROR] Profiling directory not found: {args.profiling_dir}")
        return 1

    if args.command == 'parse':
        run_msprof_parse(args.profiling_dir, args.msprof)
    elif args.command == 'export':
        run_msprof_export(args.profiling_dir, args.msprof)
    elif args.command == 'parse-and-export':
        cmd_parse_and_export(args.profiling_dir, args.msprof)
    elif args.command == 'query':
        run_msprof_query(args.profiling_dir, args.msprof)
    elif args.command == 'list':
        cmd_list(args.profiling_dir)
    elif args.command == 'summary':
        cmd_summary(args.profiling_dir)
    elif args.command == 'show':
        cmd_show_op(args.profiling_dir, args.op)
    elif args.command == 'top':
        cmd_top(args.profiling_dir, args.sort, args.top_n)
    elif args.command == 'timeline':
        cmd_timeline(args.profiling_dir)
    elif args.command == 'api':
        cmd_api(args.profiling_dir)
    elif args.command == 'plot':
        cmd_plot(args.profiling_dir, args.output)

    return 0


if __name__ == '__main__':
    sys.exit(main())
