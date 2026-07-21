#!/usr/bin/env python3
"""
Parse CANN ACL dump data files (official protobuf v2.0 format).

CANN dump files use Ascend's proprietary protobuf format and must be converted
to numpy (.npy) format using msaccucmp.py before analysis.

This script provides:
  - Auto-detection of msaccucmp.py conversion tool
  - Batch conversion of dump files to .npy
  - Analysis of converted data (list, summary, stats, compare, plot)

Usage:
    # Step 1: Convert dump files to numpy
    python parse_dump.py convert --dump_dir ./dump_data --output ./dump_data_npy

    # Step 2: Analyze converted data
    python parse_dump.py list --npy_dir ./dump_data_npy
    python parse_dump.py summary --npy_dir ./dump_data_npy
    python parse_dump.py show --npy_dir ./dump_data_npy --node "Softmax"
    python parse_dump.py compare --npy_dir ./dump_data_npy --nodes "Conv1" "Conv2"
    python parse_dump.py plot --npy_dir ./dump_data_npy --node "Softmax"

Directory structure (CANN official):
    {dump_path}/{time}/{device_id}/{model_name}/{model_id}/{data_index}/{dump_file}

Dump file naming:
    {op_type}.{op_name}.{task_id}.{stream_id}.{timestamp}
"""

import os
import sys
import argparse
import glob
import subprocess
import re
from pathlib import Path
from collections import defaultdict

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("[WARN] numpy not installed, analysis features will be limited")

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def find_msaccucmp():
    """Auto-detect msaccucmp.py location."""
    search_paths = [
        os.environ.get('ASCEND_HOME', ''),
        '/usr/local/Ascend/cann-8.5.0',
        '/usr/local/Ascend/ascend-toolkit/latest',
        '/usr/local/Ascend/latest',
    ]
    
    for base in search_paths:
        if not base:
            continue
        candidate = os.path.join(base, 'tools/operator_cmp/compare/msaccucmp.py')
        if os.path.isfile(candidate):
            return candidate
    
    return None


def parse_dump_filename(filename):
    """Parse CANN dump file name to extract metadata.
    
    Format: {op_type}.{op_name}.{task_id}.{stream_id}.{timestamp}
    Example: BatchMatMulV2._encoder_layer_0_attention_self_value_MatMul.8.46.1781687405310328
    
    Note: op_name may contain underscores but not dots. The last 3 parts are always
    task_id, stream_id, and timestamp (all numeric).
    """
    basename = os.path.basename(filename)
    parts = basename.split('.')
    
    if len(parts) < 5:
        return None
    
    # Last 3 parts are always: task_id, stream_id, timestamp
    timestamp = parts[-1]
    stream_id = parts[-2]
    task_id = parts[-3]
    
    # First part is op_type
    op_type = parts[0]
    
    # Everything between op_type and task_id is op_name
    # (may be multiple parts if op_name contains dots, though rare)
    if len(parts) == 5:
        op_name = parts[1]
    else:
        op_name = '.'.join(parts[1:-3])
    
    return {
        'op_type': op_type,
        'op_name': op_name,
        'task_id': task_id,
        'stream_id': stream_id,
        'timestamp': timestamp,
        'full_name': f"{op_type}.{op_name}"
    }


def parse_dump_path(filepath):
    """Parse full dump path to extract CANN directory metadata.
    
    Structure: {dump_path}/{time}/{device_id}/{model_name}/{model_id}/{data_index}/{dump_file}
    """
    parts = Path(filepath).parts
    
    if len(parts) < 6:
        return None
    
    dump_file = parts[-1]
    data_index = parts[-2]
    model_id = parts[-3]
    model_name = parts[-4]
    device_id = parts[-5]
    time_str = parts[-6]
    
    file_meta = parse_dump_filename(dump_file)
    if not file_meta:
        return None
    
    return {
        **file_meta,
        'time': time_str,
        'device_id': device_id,
        'model_name': model_name,
        'model_id': model_id,
        'data_index': data_index,
        'filepath': filepath
    }


def find_dump_leaf_dirs(dump_dir):
    """Find leaf directories containing dump files.
    
    CANN dump structure: {dump_path}/{time}/{device_id}/{model_name}/{model_id}/{data_index}/{files}
    msaccucmp.py only processes files in the given directory (no recursion),
    so we need to find the actual leaf directories.
    """
    leaf_dirs = []
    for root, dirs, files in os.walk(dump_dir):
        dump_files = [f for f in files if not f.endswith(('.npy', '.csv', '.txt', '.json', '.pbtxt'))]
        if dump_files:
            leaf_dirs.append(root)
    return leaf_dirs


def convert_dump_files(dump_dir, output_dir, msaccucmp_path=None):
    """Convert CANN dump files to numpy format using msaccucmp.py."""
    if not HAS_NUMPY:
        print("[ERROR] numpy is required for conversion")
        return False
    
    if msaccucmp_path is None:
        msaccucmp_path = find_msaccucmp()
    
    if not msaccucmp_path or not os.path.isfile(msaccucmp_path):
        print("[ERROR] msaccucmp.py not found. Please specify --msaccucmp or set ASCEND_HOME")
        print("        Typical location: $ASCEND_HOME/tools/operator_cmp/compare/msaccucmp.py")
        return False
    
    os.makedirs(output_dir, exist_ok=True)
    
    leaf_dirs = find_dump_leaf_dirs(dump_dir)
    
    if not leaf_dirs:
        print(f"[INFO] No dump files found in {dump_dir}")
        return True
    
    total_dump_files = 0
    for ld in leaf_dirs:
        total_dump_files += len([f for f in os.listdir(ld) 
                                 if not f.endswith(('.npy', '.csv', '.txt', '.json', '.pbtxt'))])
    
    print(f"[INFO] Found {total_dump_files} dump files in {len(leaf_dirs)} directory(ies)")
    print(f"[INFO] Converting to numpy format...")
    print(f"[INFO] Output directory: {output_dir}")
    print()
    
    for leaf_dir in leaf_dirs:
        print(f"[INFO] Processing: {leaf_dir}")
        
        cmd = [
            'python3', msaccucmp_path, 'convert',
            '-d', leaf_dir,
            '-out', output_dir,
            '-v', '2',
            '-t', 'npy'
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            
            if result.returncode != 0:
                print(f"[ERROR] Conversion failed for {leaf_dir}:")
                if result.stderr:
                    print(result.stderr[-500:])
                continue
            
            if result.stdout:
                for line in result.stdout.strip().split('\n'):
                    if '[INFO]' in line or '[ERROR]' in line:
                        print(f"  {line}")
            
        except subprocess.TimeoutExpired:
            print(f"[ERROR] Conversion timed out for {leaf_dir}")
            continue
        except Exception as e:
            print(f"[ERROR] Conversion failed for {leaf_dir}: {e}")
            continue
    
    npy_files = glob.glob(os.path.join(output_dir, '**/*.npy'), recursive=True)
    print(f"\n[INFO] Conversion complete: {len(npy_files)} .npy files generated")
    return True


def find_npy_files(npy_dir):
    """Find all .npy files in directory."""
    return sorted(glob.glob(os.path.join(npy_dir, '**/*.npy'), recursive=True))


def load_npy(filepath):
    """Load numpy file."""
    if not HAS_NUMPY:
        print(f"[ERROR] numpy required to load {filepath}")
        return None
    try:
        data = np.load(filepath, allow_pickle=False)
        return data
    except Exception as e:
        print(f"[ERROR] Failed to load {filepath}: {e}")
        return None


def extract_op_info(npy_filename):
    """Extract operator info from converted .npy filename.
    
    msaccucmp.py preserves the original dump filename and adds .input.N or .output.N suffix.
    Example: BatchMatMulV2._encoder_layer_0_attention_self_value_MatMul.8.46.1781687405310328.input.0.npy
    """
    basename = os.path.basename(npy_filename)
    if basename.endswith('.npy'):
        basename = basename[:-4]
    
    # Remove .input.N or .output.N suffix added by msaccucmp.py
    io_match = re.search(r'\.(input|output)\.\d+$', basename)
    if io_match:
        io_type = io_match.group(1)
        io_index = io_match.group(0).split('.')[-1]
        basename = basename[:io_match.start()]
    else:
        io_type = None
        io_index = None
    
    meta = parse_dump_filename(basename)
    if not meta:
        return {
            'op_type': 'Unknown',
            'op_name': basename,
            'full_name': basename,
            'io_type': io_type,
            'io_index': io_index
        }
    
    meta['io_type'] = io_type
    meta['io_index'] = io_index
    return meta


def list_operators(npy_dir):
    """List all operators in converted dump directory."""
    npy_files = find_npy_files(npy_dir)
    
    if not npy_files:
        print(f"[INFO] No .npy files found in {npy_dir}")
        print("       Run 'convert' command first to convert dump files")
        return
    
    operators = defaultdict(lambda: {'files': [], 'meta': None})
    
    for filepath in npy_files:
        meta = extract_op_info(filepath)
        op_key = meta['full_name']
        operators[op_key]['files'].append(filepath)
        if operators[op_key]['meta'] is None:
            operators[op_key]['meta'] = meta
    
    print(f"\n{'='*80}")
    print(f"Operators in {npy_dir}")
    print(f"{'='*80}")
    print(f"Total operators: {len(operators)}")
    print(f"Total files: {len(npy_files)}")
    print()
    
    op_types = defaultdict(int)
    for op_key, data in operators.items():
        op_types[data['meta']['op_type']] += 1
    
    print("Operator types:")
    for op_type in sorted(op_types.keys()):
        print(f"  {op_type:40s} {op_types[op_type]:4d} instances")
    print()
    
    print("All operators:")
    for op_key in sorted(operators.keys()):
        data = operators[op_key]
        meta = data['meta']
        files = data['files']
        print(f"  {meta['op_type']:30s} {meta['op_name']}")
        if len(files) > 1:
            print(f"    └─ {len(files)} dump files")


def summarize_dump_data(npy_dir):
    """Show summary of all converted dump data."""
    npy_files = find_npy_files(npy_dir)
    
    if not npy_files:
        print(f"[INFO] No .npy files found in {npy_dir}")
        return
    
    print(f"\n{'='*80}")
    print(f"Dump Data Summary: {npy_dir}")
    print(f"{'='*80}")
    print(f"Total files: {len(npy_files)}")
    print()
    
    operators = defaultdict(lambda: {'files': [], 'meta': None})
    for filepath in npy_files:
        meta = extract_op_info(filepath)
        op_key = meta['full_name']
        operators[op_key]['files'].append(filepath)
        if operators[op_key]['meta'] is None:
            operators[op_key]['meta'] = meta
    
    print(f"Total operators: {len(operators)}")
    print()
    
    for op_key in sorted(operators.keys())[:20]:
        data = operators[op_key]
        meta = data['meta']
        files = data['files']
        print(f"Operator: {meta['op_type']}.{meta['op_name']}")
        print(f"  Files: {len(files)}")
        
        for filepath in files[:3]:
            npy_data = load_npy(filepath)
            if npy_data is not None:
                print(f"    - shape={npy_data.shape}, dtype={npy_data.dtype}, size={npy_data.nbytes} bytes")
        
        if len(files) > 3:
            print(f"    ... and {len(files) - 3} more files")
        print()
    
    if len(operators) > 20:
        print(f"... and {len(operators) - 20} more operators")


def show_operator_data(npy_dir, node_name, show_stats=True):
    """Show detailed data for a specific operator."""
    npy_files = find_npy_files(npy_dir)
    
    matching_files = [f for f in npy_files if node_name in os.path.basename(f)]
    
    if not matching_files:
        print(f"[INFO] No files found for operator: {node_name}")
        print(f"       Try: python parse_dump.py list --npy_dir {npy_dir}")
        return
    
    print(f"\n{'='*80}")
    print(f"Operator Data: {node_name}")
    print(f"{'='*80}")
    print(f"Matching files: {len(matching_files)}")
    print()
    
    for filepath in matching_files:
        meta = extract_op_info(filepath)
        print(f"[{meta['op_type']}] {os.path.basename(filepath)}")
        
        data = load_npy(filepath)
        if data is None:
            continue
        
        print(f"  Shape: {data.shape}")
        print(f"  Dtype: {data.dtype}")
        print(f"  Size:  {data.nbytes} bytes")
        
        if show_stats and data.size > 0:
            # Convert to float32 for accurate statistics (float16 can produce nan)
            stats_data = data.astype(np.float32) if data.dtype == np.float16 else data
            # Use nan-aware functions to handle NaN values in data
            print(f"  Min:   {np.nanmin(stats_data)}")
            print(f"  Max:   {np.nanmax(stats_data)}")
            print(f"  Mean:  {np.nanmean(stats_data)}")
            print(f"  Std:   {np.nanstd(stats_data)}")
            
            nan_count = np.isnan(stats_data).sum()
            if nan_count > 0:
                print(f"  NaN:   {nan_count} values ({nan_count/stats_data.size*100:.2f}%)")
            
            if data.size <= 20:
                print(f"  Values: {data.flatten()}")
            else:
                print(f"  First 10 values: {data.flatten()[:10]}")
        
        print()


def plot_operator_data(npy_dir, node_name, output_file=None):
    """Plot operator data distribution."""
    if not HAS_MATPLOTLIB:
        print("[ERROR] matplotlib required for plotting")
        return
    
    npy_files = find_npy_files(npy_dir)
    matching_files = [f for f in npy_files if node_name in os.path.basename(f)]
    
    if not matching_files:
        print(f"[INFO] No files found for operator: {node_name}")
        return
    
    fig, axes = plt.subplots(len(matching_files), 1, figsize=(12, 4 * len(matching_files)))
    if len(matching_files) == 1:
        axes = [axes]
    
    for idx, filepath in enumerate(matching_files):
        data = load_npy(filepath)
        if data is None:
            continue
        
        data_flat = data.flatten()
        ax = axes[idx]
        ax.hist(data_flat, bins=100, alpha=0.7, edgecolor='black')
        ax.set_title(f"{os.path.basename(filepath)} - shape={data.shape}")
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"[INFO] Plot saved to: {output_file}")
    else:
        plt.show()


def compare_tensors(t1, t2, label1="Tensor 1", label2="Tensor 2"):
    """Core tensor comparison logic with NaN-aware statistics."""
    if t1.shape != t2.shape:
        print(f"[WARN] Shapes differ ({t1.shape} vs {t2.shape}), flattening for comparison")
        t1 = t1.flatten()
        t2 = t2.flatten()
        min_len = min(len(t1), len(t2))
        t1 = t1[:min_len]
        t2 = t2[:min_len]
    else:
        t1 = t1.flatten()
        t2 = t2.flatten()
    
    if len(t1) == 0 or len(t2) == 0:
        print("[ERROR] Empty tensors, cannot compare")
        return
    
    # Convert to float32 for accurate comparison
    t1 = t1.astype(np.float32)
    t2 = t2.astype(np.float32)
    
    # Filter out NaN values for comparison
    valid_mask = ~(np.isnan(t1) | np.isnan(t2))
    t1_valid = t1[valid_mask]
    t2_valid = t2[valid_mask]
    
    nan_count = (~valid_mask).sum()
    if nan_count > 0:
        print(f"[INFO] Ignoring {nan_count} NaN values ({nan_count/len(t1)*100:.2f}%)")
        print()
    
    if len(t1_valid) == 0:
        print("[ERROR] All values are NaN, cannot compare")
        return
    
    # Basic stats for each tensor
    print(f"{label1}: min={np.nanmin(t1):.6g}, max={np.nanmax(t1):.6g}, mean={np.nanmean(t1):.6g}")
    print(f"{label2}: min={np.nanmin(t2):.6g}, max={np.nanmax(t2):.6g}, mean={np.nanmean(t2):.6g}")
    print()
    
    diff = np.abs(t1_valid - t2_valid)
    print(f"Absolute Difference:")
    print(f"  Min:  {diff.min()}")
    print(f"  Max:  {diff.max()}")
    print(f"  Mean: {diff.mean()}")
    print(f"  Std:  {diff.std()}")
    print()
    
    rel_diff = diff / (np.abs(t1_valid) + 1e-8)
    print(f"Relative Difference:")
    print(f"  Min:  {rel_diff.min()}")
    print(f"  Max:  {rel_diff.max()}")
    print(f"  Mean: {rel_diff.mean()}")
    print()
    
    cos_sim = np.dot(t1_valid, t2_valid) / (np.linalg.norm(t1_valid) * np.linalg.norm(t2_valid) + 1e-8)
    print(f"Cosine Similarity: {cos_sim:.8f}")


def compare_operators(npy_dir, node1, node2):
    """Compare two operator tensors within the same directory."""
    if not HAS_NUMPY:
        print("[ERROR] numpy required for comparison")
        return
    
    npy_files = find_npy_files(npy_dir)
    
    files1 = [f for f in npy_files if node1 in os.path.basename(f)]
    files2 = [f for f in npy_files if node2 in os.path.basename(f)]
    
    if not files1:
        print(f"[ERROR] No files found for: {node1}")
        return
    if not files2:
        print(f"[ERROR] No files found for: {node2}")
        return
    
    data1 = load_npy(files1[0])
    data2 = load_npy(files2[0])
    
    if data1 is None or data2 is None:
        print("[ERROR] Failed to load tensor data")
        return
    
    print(f"\n{'='*80}")
    print(f"Operator Comparison: {node1} vs {node2}")
    print(f"{'='*80}")
    print(f"Operator 1: shape={data1.shape}, dtype={data1.dtype}, file={os.path.basename(files1[0])}")
    print(f"Operator 2: shape={data2.shape}, dtype={data2.dtype}, file={os.path.basename(files2[0])}")
    print()
    
    compare_tensors(data1, data2, label1="Operator 1", label2="Operator 2")


def diff_directories(npy_dir1, node1, npy_dir2, node2):
    """Compare operator tensors from two different directories."""
    if not HAS_NUMPY:
        print("[ERROR] numpy required for comparison")
        return
    
    files1 = find_npy_files(npy_dir1)
    files2 = find_npy_files(npy_dir2)
    
    matches1 = [f for f in files1 if node1 in os.path.basename(f)]
    matches2 = [f for f in files2 if node2 in os.path.basename(f)]
    
    if not matches1:
        print(f"[ERROR] No files found for '{node1}' in {npy_dir1}")
        return
    if not matches2:
        print(f"[ERROR] No files found for '{node2}' in {npy_dir2}")
        return
    
    if len(matches1) > 1:
        print(f"[INFO] Multiple matches for '{node1}' in dir1, using first: {os.path.basename(matches1[0])}")
    if len(matches2) > 1:
        print(f"[INFO] Multiple matches for '{node2}' in dir2, using first: {os.path.basename(matches2[0])}")
    
    data1 = load_npy(matches1[0])
    data2 = load_npy(matches2[0])
    
    if data1 is None or data2 is None:
        print("[ERROR] Failed to load tensor data")
        return
    
    print(f"\n{'='*80}")
    print(f"Cross-Directory Comparison")
    print(f"{'='*80}")
    print(f"Dir1: {npy_dir1}")
    print(f"  Node: {node1}")
    print(f"  File: {os.path.basename(matches1[0])}")
    print(f"  Shape: {data1.shape}, dtype={data1.dtype}")
    print()
    print(f"Dir2: {npy_dir2}")
    print(f"  Node: {node2}")
    print(f"  File: {os.path.basename(matches2[0])}")
    print(f"  Shape: {data2.shape}, dtype={data2.dtype}")
    print()
    
    compare_tensors(data1, data2, label1="Dir1", label2="Dir2")


def main():
    parser = argparse.ArgumentParser(
        description='Parse CANN ACL dump data (protobuf v2.0 format)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert dump files to numpy
  python parse_dump.py convert --dump_dir ./dump_data --output ./dump_data_npy

  # List all operators
  python parse_dump.py list --npy_dir ./dump_data_npy

  # Show summary
  python parse_dump.py summary --npy_dir ./dump_data_npy

  # Show specific operator data
  python parse_dump.py show --npy_dir ./dump_data_npy --node "Softmax"

  # Compare two operators in the same directory
  python parse_dump.py compare --npy_dir ./dump_data_npy --nodes "Conv1" "Conv2"

  # Compare operators from two different directories
  python parse_dump.py diff --npy_dir1 ./dump1_npy --node1 "Softmax" \
                            --npy_dir2 ./dump2_npy --node2 "Softmax"

  # Plot operator data
  python parse_dump.py plot --npy_dir ./dump_data_npy --node "Softmax"
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    convert_parser = subparsers.add_parser('convert', help='Convert dump files to numpy format')
    convert_parser.add_argument('--dump_dir', type=str, default='./dump_data',
                                help='Directory containing CANN dump files')
    convert_parser.add_argument('--output', type=str, default='./dump_data_npy',
                                help='Output directory for .npy files')
    convert_parser.add_argument('--msaccucmp', type=str, default=None,
                                help='Path to msaccucmp.py (auto-detected if not specified)')
    
    list_parser = subparsers.add_parser('list', help='List all operators')
    list_parser.add_argument('--npy_dir', type=str, default='./dump_data_npy',
                             help='Directory containing converted .npy files')
    
    summary_parser = subparsers.add_parser('summary', help='Show summary of dump data')
    summary_parser.add_argument('--npy_dir', type=str, default='./dump_data_npy',
                                help='Directory containing converted .npy files')
    
    show_parser = subparsers.add_parser('show', help='Show detailed operator data')
    show_parser.add_argument('--npy_dir', type=str, default='./dump_data_npy',
                             help='Directory containing converted .npy files')
    show_parser.add_argument('--node', type=str, required=True,
                             help='Operator name (partial match supported)')
    show_parser.add_argument('--no_stats', action='store_true',
                             help='Disable statistics display')
    
    compare_parser = subparsers.add_parser('compare', help='Compare two operators in the same directory')
    compare_parser.add_argument('--npy_dir', type=str, default='./dump_data_npy',
                                help='Directory containing converted .npy files')
    compare_parser.add_argument('--nodes', nargs=2, metavar=('NODE1', 'NODE2'), required=True,
                                help='Two operator names to compare')
    
    diff_parser = subparsers.add_parser('diff', help='Compare operators from two different directories')
    diff_parser.add_argument('--npy_dir1', type=str, required=True,
                             help='First directory containing converted .npy files')
    diff_parser.add_argument('--node1', type=str, required=True,
                             help='Operator name in first directory (partial match supported)')
    diff_parser.add_argument('--npy_dir2', type=str, required=True,
                             help='Second directory containing converted .npy files')
    diff_parser.add_argument('--node2', type=str, required=True,
                             help='Operator name in second directory (partial match supported)')
    
    plot_parser = subparsers.add_parser('plot', help='Plot operator data distribution')
    plot_parser.add_argument('--npy_dir', type=str, default='./dump_data_npy',
                             help='Directory containing converted .npy files')
    plot_parser.add_argument('--node', type=str, required=True,
                             help='Operator name (partial match supported)')
    plot_parser.add_argument('--output', type=str, default=None,
                             help='Save plot to file instead of displaying')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    if args.command == 'convert':
        if not os.path.exists(args.dump_dir):
            print(f"[ERROR] Dump directory not found: {args.dump_dir}")
            return 1
        
        success = convert_dump_files(args.dump_dir, args.output, args.msaccucmp)
        return 0 if success else 1
    
    elif args.command == 'list':
        if not os.path.exists(args.npy_dir):
            print(f"[ERROR] NPY directory not found: {args.npy_dir}")
            print("       Run 'convert' command first")
            return 1
        list_operators(args.npy_dir)
        return 0
    
    elif args.command == 'summary':
        if not os.path.exists(args.npy_dir):
            print(f"[ERROR] NPY directory not found: {args.npy_dir}")
            return 1
        summarize_dump_data(args.npy_dir)
        return 0
    
    elif args.command == 'show':
        if not os.path.exists(args.npy_dir):
            print(f"[ERROR] NPY directory not found: {args.npy_dir}")
            return 1
        show_operator_data(args.npy_dir, args.node, show_stats=not args.no_stats)
        return 0
    
    elif args.command == 'compare':
        if not os.path.exists(args.npy_dir):
            print(f"[ERROR] NPY directory not found: {args.npy_dir}")
            return 1
        compare_operators(args.npy_dir, args.nodes[0], args.nodes[1])
        return 0
    
    elif args.command == 'diff':
        if not os.path.exists(args.npy_dir1):
            print(f"[ERROR] NPY directory not found: {args.npy_dir1}")
            return 1
        if not os.path.exists(args.npy_dir2):
            print(f"[ERROR] NPY directory not found: {args.npy_dir2}")
            return 1
        diff_directories(args.npy_dir1, args.node1, args.npy_dir2, args.node2)
        return 0
    
    elif args.command == 'plot':
        if not os.path.exists(args.npy_dir):
            print(f"[ERROR] NPY directory not found: {args.npy_dir}")
            return 1
        plot_operator_data(args.npy_dir, args.node, args.output)
        return 0
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
