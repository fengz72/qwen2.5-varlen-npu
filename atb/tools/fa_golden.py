import numpy as np
from typing import Tuple, Optional

class FlashAttentionGolden:
    """
    FlashAttention的Golden实现，使用numpy模拟分块计算过程
    用于作为精度基准与其他平台(GPU/NPU)的输出进行对比
    """
    
    def __init__(self, softmax_scale: Optional[float] = None, block_size: int = 64):
        """
        初始化FlashAttention Golden
        
        Args:
            softmax_scale: softmax的缩放因子，默认1/sqrt(d)
            block_size: 分块大小，模拟FlashAttention的分块计算
        """
        self.softmax_scale = softmax_scale
        self.block_size = block_size
    
    def _standard_attention(self, q: np.ndarray, k: np.ndarray, v: np.ndarray, 
                           atten_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        标准attention计算（非分块），用于验证分块计算的正确性
        
        Args:
            q: query tensor, shape (B, S, N, D)
            k: key tensor, shape (B, S, N, D)
            v: value tensor, shape (B, S, N, D)
            atten_mask: attention mask, shape (B, 1, S, S) or (1, 1, S, S) or (S, S)
        
        Returns:
            output: shape (B, S, N, D)
        """
        B, S, N, D = q.shape
        
        if self.softmax_scale is None:
            self.softmax_scale = 1.0 / np.sqrt(D)
        
        # Q * K^T: (B, S, N, D) @ (B, S, N, D) -> (B, N, S, S)
        q_reshaped = q.transpose(0, 2, 1, 3)  # (B, N, S, D)
        k_reshaped = k.transpose(0, 2, 1, 3)  # (B, N, S, D)
        
        scores = np.matmul(q_reshaped.astype(np.float32), k_reshaped.astype(np.float32).transpose(0, 1, 3, 2)).astype(np.float16).astype(np.float32)  # (B, N, S, S)
        scores = scores * self.softmax_scale
        
        # 应用attention mask
        if atten_mask is not None:
            # 扩展mask维度以匹配scores
            if atten_mask.ndim == 2:
                atten_mask = atten_mask[np.newaxis, np.newaxis, :, :]
            elif atten_mask.ndim == 3:
                atten_mask = atten_mask[:, np.newaxis, :, :]
            scores = scores + atten_mask
        
        # Softmax
        scores_max = np.max(scores, axis=-1, keepdims=True)
        scores_exp = np.exp(scores - scores_max)
        scores_sum = np.sum(scores_exp, axis=-1, keepdims=True)
        atten_weights = scores_exp / (scores_sum + 1e-8)
        
        # Weighted sum: (B, N, S, S) @ (B, N, S, D) -> (B, N, S, D)
        v_reshaped = v.transpose(0, 2, 1, 3)  # (B, N, S, D)
        output = np.matmul(atten_weights.astype(np.float32), v_reshaped.astype(np.float32)).astype(np.float16).astype(np.float32)  # (B, N, S, D)
        output = output.transpose(0, 2, 1, 3)  # (B, S, N, D)
        
        return output
    
    def _flash_attention_tiled(self, q: np.ndarray, k: np.ndarray, v: np.ndarray,
                               atten_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        使用分块计算的FlashAttention实现
        模拟GPU上FlashAttention的分块计算逻辑
        
        Args:
            q: query tensor, shape (B, S, N, D)
            k: key tensor, shape (B, S, N, D)
            v: value tensor, shape (B, S, N, D)
            atten_mask: attention mask
        
        Returns:
            output: shape (B, S, N, D)
        """
        B, S, N, D = q.shape
        
        if self.softmax_scale is None:
            self.softmax_scale = 1.0 / np.sqrt(D)
        
        # 初始化输出和辅助变量
        output = np.zeros((B, S, N, D), dtype=np.float32)
        
        # 外层循环：遍历query的分块
        for b in range(B):
            for n in range(N):
                # 获取当前head的Q, K, V
                q_bn = q[b, :, n, :]  # (S, D)
                k_bn = k[b, :, n, :]  # (S, D)
                v_bn = v[b, :, n, :]  # (S, D)
                
                # 内层循环：对query序列进行分块
                for q_start in range(0, S, self.block_size):
                    q_end = min(q_start + self.block_size, S)
                    q_block = q_bn[q_start:q_end]  # (Br, D)
                    
                    # 初始化online softmax的状态
                    m_i = np.full((q_end - q_start, 1), -np.inf)  # (Br, 1)
                    l_i = np.zeros((q_end - q_start, 1))  # (Br, 1)
                    o_i = np.zeros((q_end - q_start, D))  # (Br, D)
                    
                    # 遍历key/value的分块
                    for k_start in range(0, S, self.block_size):
                        k_end = min(k_start + self.block_size, S)
                        k_block = k_bn[k_start:k_end]  # (Bc, D)
                        v_block = v_bn[k_start:k_end]  # (Bc, D)
                        
                        # 计算当前分块的注意力分数
                        s_ij = np.dot(q_block, k_block.T) * self.softmax_scale  # (Br, Bc)
                        
                        # 应用mask
                        if atten_mask is not None:
                            mask_slice = self._get_mask_slice(atten_mask, b, q_start, q_end, k_start, k_end)
                            s_ij = s_ij + mask_slice
                        
                        # Online softmax更新
                        m_ij = np.max(s_ij, axis=1, keepdims=True)  # (Br, 1)
                        p_ij = np.exp(s_ij - m_ij)  # (Br, Bc)
                        l_ij = np.sum(p_ij, axis=1, keepdims=True)  # (Br, 1)
                        
                        # 更新running max和sum
                        m_new = np.maximum(m_i, m_ij)
                        alpha = np.exp(m_i - m_new)
                        beta = np.exp(m_ij - m_new)
                        
                        # 更新累加器
                        l_i = alpha * l_i + beta * l_ij
                        o_i = alpha * o_i + beta * np.dot(p_ij, v_block)
                        m_i = m_new
                    
                    # 最终归一化
                    o_i = o_i / (l_i + 1e-8)
                    output[b, q_start:q_end, n, :] = o_i
        
        return output
    
    def _get_mask_slice(self, atten_mask: np.ndarray, batch_idx: int, 
                        q_start: int, q_end: int, k_start: int, k_end: int) -> np.ndarray:
        """获取mask的对应分块"""
        if atten_mask.ndim == 4:
            # (B, 1, S, S) or (B, N, S, S)
            mask_slice = atten_mask[batch_idx, 0, q_start:q_end, k_start:k_end]
        elif atten_mask.ndim == 3:
            # (1, S, S)
            mask_slice = atten_mask[0, q_start:q_end, k_start:k_end]
        else:
            # (S, S)
            mask_slice = atten_mask[q_start:q_end, k_start:k_end]
        return mask_slice.astype(np.float32)
    
    def forward(self, q: np.ndarray, k: np.ndarray, v: np.ndarray, 
                atten_mask: Optional[np.ndarray] = None, 
                use_tiled: bool = True) -> np.ndarray:
        """
        前向计算
        
        Args:
            q: query, shape (B, S, N, D)
            k: key, shape (B, S, N, D)  
            v: value, shape (B, S, N, D)
            atten_mask: attention mask
            use_tiled: 是否使用分块计算（True: FlashAttention方式, False: 标准attention）
        
        Returns:
            output: shape (B, S, N, D)
        """
        if use_tiled:
            return self._flash_attention_tiled(q, k, v, atten_mask)
        else:
            return self._standard_attention(q, k, v, atten_mask)


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
            print(f"形状: {golden.shape}")
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


# 使用示例
if __name__ == "__main__":
    print("FlashAttention Golden 使用示例")
    print("="*60)
    
    # 设置随机种子以确保可重现性
    np.random.seed(42)
    
    # 定义输入维度 B=2, S=128, N=4, D=64
    B, S, N, D = 1, 114, 12, 64
    
    # 生成测试输入数据
    q = np.random.randn(B, S, N, D).astype(np.float32) * 0.1
    k = np.random.randn(B, S, N, D).astype(np.float32) * 0.1
    v = np.random.randn(B, S, N, D).astype(np.float32) * 0.1
    q = np.load('dump_data_npy/FlashAttentionScore.FusedFlashAttention_PartitionedCall__encoder_layer_0_attention_self_Transpose_3_Transpose_163.80.46.1781603802039360.input.0.npy')
    k = np.load('dump_data_npy/FlashAttentionScore.FusedFlashAttention_PartitionedCall__encoder_layer_0_attention_self_Transpose_3_Transpose_163.80.46.1781603802039360.input.1.npy')
    v = np.load('dump_data_npy/FlashAttentionScore.FusedFlashAttention_PartitionedCall__encoder_layer_0_attention_self_Transpose_3_Transpose_163.80.46.1781603802039360.input.2.npy')
    
    # 创建causal mask（下三角mask，上三角屏蔽）
    causal_mask = np.triu(np.ones((S, S)) * -1e10, k=1)
    causal_mask = np.load('dump_data_npy/FlashAttentionScore.FusedFlashAttention_PartitionedCall__encoder_layer_0_attention_self_Transpose_3_Transpose_163.80.46.1781603802039360.input.3.npy')
    
    # 初始化FlashAttention Golden
    fa_golden = FlashAttentionGolden(block_size=32)
    
    # 计算golden输出（使用分块计算）
    golden_output = fa_golden.forward(q, k, v, atten_mask=causal_mask, use_tiled=True)
    print(f"Golden输出形状: {golden_output.shape}")
    print(f"Golden输出统计: mean={np.mean(golden_output):.6f}, std={np.std(golden_output):.6f}")
    
    # 验证分块计算与标准计算的一致性
    standard_output = fa_golden.forward(q, k, v, atten_mask=causal_mask, use_tiled=False)
    
    npu_out = np.load('dump_data_npy/FlashAttentionScore.FusedFlashAttention_PartitionedCall__encoder_layer_0_attention_self_Transpose_3_Transpose_163.80.46.1781603802039360.output.3.npy')

    print(standard_output)
    print(standard_output.shape)
    print(npu_out)
    print(npu_out.shape)
    metrics = PrecisionComparator.compare_and_report(
        standard_output, golden_output, 
        target_name="FlashAttention(Tiled)",
        rtol=1e-3, atol=1e-3
    )
    
    # 访问具体的指标
    print("关键指标访问示例:")
    print(f"相对误差超过rtol的占比: {metrics['rel_error_exceed_rtol_ratio']*100:.4f}%")
    print(f"相对误差超过rtol的点数: {metrics['rel_error_exceed_rtol_count']:,}")
    print(f"绝对误差超过atol的占比: {metrics['abs_error_exceed_atol_ratio']*100:.4f}%")
    print(f"绝对误差超过atol的点数: {metrics['abs_error_exceed_atol_count']:,}")
