import argparse
import os
import sys
import numpy as np


class PrecisionComparator:
    """
    精度对比工具，用于比较不同实现的输出
    """
    
    @staticmethod
    def compute_metrics(golden: np.ndarray, target: np.ndarray, 
                        rtol: float = 1e-3, atol: float = 1e-5) -> dict:
        """
        计算golden和target之间的各种精度指标
        
        Args:
            golden: golden参考输出
            target: 待比较的输出
            rtol: 相对误差容忍度
            atol: 绝对误差容忍度
        
        Returns:
            metrics: 包含各种精度指标的字典
        """
        # 确保数据类型一致
        golden = golden.astype(np.float32)
        target = target.astype(np.float32)
        
        # 绝对误差
        abs_diff = np.abs(golden - target)
        
        # 相对误差（避免除零）
        denom = np.maximum(np.abs(golden), 1e-8)
        rel_diff = abs_diff / denom
        
        # 总元素数量
        total_elements = golden.size
        
        # 计算各种指标
        metrics = {
            'max_abs_error': float(np.max(abs_diff)),
            'mean_abs_error': float(np.mean(abs_diff)),
            'max_rel_error': float(np.max(rel_diff)),
            'mean_rel_error': float(np.mean(rel_diff)),
            'rmse': float(np.sqrt(np.mean((golden - target) ** 2))),
            'cosine_similarity': float(PrecisionComparator._cosine_sim(golden, target)),
            'pearson_corr': float(PrecisionComparator._pearson_corr(golden, target)),
            'relative_l2_error': float(np.linalg.norm(golden - target) / (np.linalg.norm(golden) + 1e-8)),
            'total_elements': total_elements,
        }
        
        # 检查是否在容忍度内
        metrics['pass_rtol_atol'] = bool(np.allclose(golden, target, rtol=rtol, atol=atol))
        metrics['max_diff_position'] = np.unravel_index(np.argmax(abs_diff), abs_diff.shape)
        
        # 统计相对误差超过rtol的点数占比
        exceed_count = np.sum(rel_diff > rtol)
        metrics['rel_error_exceed_rtol_count'] = int(exceed_count)
        metrics['rel_error_exceed_rtol_ratio'] = float(exceed_count) / total_elements if total_elements > 0 else 0.0
        
        # 同时统计绝对误差超过atol的点数占比
        exceed_atol_count = np.sum(abs_diff > atol)
        metrics['abs_error_exceed_atol_count'] = int(exceed_atol_count)
        metrics['abs_error_exceed_atol_ratio'] = float(exceed_atol_count) / total_elements if total_elements > 0 else 0.0
        
        # 统计误差分布
        error_percentiles = [50, 90, 95, 99, 99.9]
        for p in error_percentiles:
            metrics[f'abs_error_p{p}'] = float(np.percentile(abs_diff, p))
            metrics[f'rel_error_p{p}'] = float(np.percentile(rel_diff, p))
        
        return metrics
    
    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        """计算余弦相似度"""
        a_flat = a.flatten()
        b_flat = b.flatten()
        return np.dot(a_flat, b_flat) / (np.linalg.norm(a_flat) * np.linalg.norm(b_flat) + 1e-8)
    
    @staticmethod
    def _pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
        """计算Pearson相关系数"""
        a_flat = a.flatten()
        b_flat = b.flatten()
        a_centered = a_flat - np.mean(a_flat)
        b_centered = b_flat - np.mean(b_flat)
        return np.dot(a_centered, b_centered) / (
            np.sqrt(np.sum(a_centered ** 2)) * np.sqrt(np.sum(b_centered ** 2)) + 1e-8
        )
    
    @staticmethod
    def compare_and_report(golden: np.ndarray, target: np.ndarray, 
                          target_name: str = "Target",
                          rtol: float = 1e-3, atol: float = 1e-5,
                          verbose: bool = True) -> dict:
        """
        对比并打印报告
        
        Args:
            golden: golden参考输出
            target: 待比较的输出
            target_name: 目标平台名称
            rtol: 相对误差容忍度
            atol: 绝对误差容忍度
            verbose: 是否打印详细信息
        
        Returns:
            metrics: 精度指标字典
        """
        metrics = PrecisionComparator.compute_metrics(golden, target, rtol, atol)
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"精度对比报告: Golden vs {target_name}")
            print(f"{'='*60}")
            print(f"形状: {golden.shape} {target.shape}")
            print(f"数据类型: golden={golden.dtype}, target={target.dtype}")
            print(f"总元素数: {metrics['total_elements']:,}")
            print(f"\n基本统计:")
            print(f"  Golden - mean: {np.mean(golden):.6e}, std: {np.std(golden):.6e}")
            print(f"  Target - mean: {np.mean(target):.6e}, std: {np.std(target):.6e}")
            print(f"\n误差指标:")
            print(f"  最大绝对误差: {metrics['max_abs_error']:.6e}")
            print(f"  平均绝对误差: {metrics['mean_abs_error']:.6e}")
            print(f"  最大相对误差: {metrics['max_rel_error']:.6e}")
            print(f"  平均相对误差: {metrics['mean_rel_error']:.6e}")
            print(f"  RMSE: {metrics['rmse']:.6e}")
            print(f"  相对L2误差: {metrics['relative_l2_error']:.6e}")
            print(f"\n相似度指标:")
            print(f"  余弦相似度: {metrics['cosine_similarity']:.8f}")
            print(f"  Pearson相关系数: {metrics['pearson_corr']:.8f}")
            print(f"\n误差容忍度统计 (rtol={rtol}, atol={atol}):")
            print(f"  相对误差超过rtol的点数: {metrics['rel_error_exceed_rtol_count']:,} ({metrics['rel_error_exceed_rtol_ratio']*100:.4f}%)")
            print(f"  绝对误差超过atol的点数: {metrics['abs_error_exceed_atol_count']:,} ({metrics['abs_error_exceed_atol_ratio']*100:.4f}%)")
            print(f"\n误差分布:")
            for key, value in metrics.items():
                if 'error_p' in key and 'exceed' not in key:
                    print(f"  {key}: {value:.6e}")
            print(f"\n最大误差位置: {metrics['max_diff_position']}")
            print(f"\n通过测试 (rtol={rtol}, atol={atol}): {metrics['pass_rtol_atol']}")
            print(f"{'='*60}\n")
        
        return metrics


import argparse
import os
import sys
import numpy as np


DTYPE_MAP = {
    'float16': np.float16,
    'float32': np.float32,
    'float64': np.float64,
    'float':   np.float32,
    'int8':    np.int8,
    'int16':   np.int16,
    'int32':   np.int32,
    'int64':   np.int64,
    'uint8':   np.uint8,
    'uint16':  np.uint16,
    'uint32':  np.uint32,
    'uint64':  np.uint64,
    'bool':    np.bool_,
}


def load_file(filepath, dtype_str='float16', shape=None):
    """Load a .bin or .npy file and return numpy array."""
    if not os.path.exists(filepath):
        print(f"[ERROR] File not found: {filepath}")
        sys.exit(1)

    ext = os.path.splitext(filepath)[1].lower()

    if ext == '.npy':
        data = np.load(filepath, allow_pickle=False)
        print(f"[INFO] Loaded .npy: {filepath}, shape={data.shape}, dtype={data.dtype}")
    elif ext == '.bin':
        np_dtype = DTYPE_MAP.get(dtype_str)
        if np_dtype is None:
            print(f"[ERROR] Unsupported dtype: {dtype_str}")
            print(f"        Supported: {', '.join(sorted(DTYPE_MAP.keys()))}")
            sys.exit(1)
        data = np.fromfile(filepath, dtype=np_dtype)
        print(f"[INFO] Loaded .bin: {filepath}, dtype={dtype_str}, elements={data.size}")
    else:
        print(f"[ERROR] Unsupported file format: {ext} (use .bin or .npy)")
        sys.exit(1)

    if shape is not None:
        try:
            data = data.reshape(shape)
            print(f"[INFO] Reshaped to: {data.shape}")
        except ValueError as e:
            print(f"[ERROR] Cannot reshape to {shape}: {e}")
            sys.exit(1)

    return data


def main():
    parser = argparse.ArgumentParser(
        description='Compare two tensor files (.bin or .npy) for precision analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare two .npy files
  python compare.py golden.npy target.npy

  # Compare two .bin files (default dtype: float16)
  python compare.py golden.bin target.bin

  # Compare .bin files with specific dtype
  python compare.py golden.bin target.bin --dtype float32

  # Compare with different dtypes for each file
  python compare.py golden.bin target.npy --dtype1 float32 --dtype2 float16

  # Compare with reshape
  python compare.py golden.bin target.bin --shape1 80,5 --shape2 80,5

  # Compare with custom tolerance
  python compare.py golden.npy target.npy --rtol 1e-2 --atol 1e-2
        """
    )

    parser.add_argument('file1', help='First file path (golden/reference), .bin or .npy')
    parser.add_argument('file2', help='Second file path (target), .bin or .npy')
    parser.add_argument('--dtype', type=str, default='float16',
                        help='Data type for both .bin files (default: float16). '
                             'Ignored for .npy files. '
                             'Supported: float16, float32, float64, int8, int16, int32, int64, uint8, uint16, uint32, uint64, bool')
    parser.add_argument('--dtype1', type=str, default=None,
                        help='Data type for file1 only (overrides --dtype for file1)')
    parser.add_argument('--dtype2', type=str, default=None,
                        help='Data type for file2 only (overrides --dtype for file2)')
    parser.add_argument('--shape1', type=str, default=None,
                        help='Reshape file1, comma-separated (e.g. 80,5)')
    parser.add_argument('--shape2', type=str, default=None,
                        help='Reshape file2, comma-separated (e.g. 80,5)')
    parser.add_argument('--rtol', type=float, default=1e-3,
                        help='Relative tolerance (default: 1e-3)')
    parser.add_argument('--atol', type=float, default=1e-5,
                        help='Absolute tolerance (default: 1e-5)')

    args = parser.parse_args()

    dtype1 = args.dtype1 if args.dtype1 else args.dtype
    dtype2 = args.dtype2 if args.dtype2 else args.dtype

    shape1 = tuple(int(x) for x in args.shape1.split(',')) if args.shape1 else None
    shape2 = tuple(int(x) for x in args.shape2.split(',')) if args.shape2 else None

    print(f"\n{'='*60}")
    print(f"Loading files")
    print(f"{'='*60}")
    data1 = load_file(args.file1, dtype1, shape1)
    data2 = load_file(args.file2, dtype2, shape2)

    if data1.shape != data2.shape:
        print(f"\n[WARN] Shapes differ: {data1.shape} vs {data2.shape}, flattening for comparison")
        data1 = data1.flatten()
        data2 = data2.flatten()
        min_len = min(len(data1), len(data2))
        if len(data1) != len(data2):
            print(f"[WARN] Lengths differ: {len(data1)} vs {len(data2)}, truncating to {min_len}")
        data1 = data1[:min_len]
        data2 = data2[:min_len]

    PrecisionComparator.compare_and_report(
        data1, data2,
        target_name=os.path.basename(args.file2),
        rtol=args.rtol,
        atol=args.atol
    )


if __name__ == '__main__':
    main()
