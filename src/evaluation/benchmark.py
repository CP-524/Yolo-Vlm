"""基准测试工具"""
import time
import logging
from typing import Dict, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class Benchmark:
    """性能基准测试"""
    
    def __init__(self):
        """初始化基准测试"""
        self.results = []
    
    def measure_latency(self, func, *args, **kwargs) -> Dict:
        """
        测量函数延迟
        
        Args:
            func: 要测量的函数
            *args: 函数参数
            **kwargs: 函数关键字参数
            
        Returns:
            包含延迟信息的字典
        """
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        latency = end_time - start_time
        
        return {
            'result': result,
            'latency': latency,
            'timestamp': start_time
        }
    
    def benchmark_detection(
        self,
        detection_agent,
        image_paths: List[str],
        use_agentic: bool = True
    ) -> Dict:
        """
        基准测试检测性能
        
        Args:
            detection_agent: 检测Agent
            image_paths: 图像路径列表
            use_agentic: 是否使用Agentic方法
            
        Returns:
            基准测试结果
        """
        logger.info(f"Benchmarking {len(image_paths)} images")
        
        latencies = []
        
        for img_path in image_paths:
            if use_agentic:
                result = self.measure_latency(detection_agent.detect, img_path)
            else:
                result = self.measure_latency(detection_agent.baseline_detect, img_path)
            
            latencies.append(result['latency'])
        
        import numpy as np
        return {
            'mean_latency': np.mean(latencies),
            'std_latency': np.std(latencies),
            'min_latency': np.min(latencies),
            'max_latency': np.max(latencies),
            'total_time': np.sum(latencies),
            'fps': len(image_paths) / np.sum(latencies)
        }