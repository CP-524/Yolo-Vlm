"""并行处理工具"""
import concurrent.futures
from typing import List, Callable, Any
import logging

logger = logging.getLogger(__name__)


def parallel_process(
    items: List[Any],
    process_func: Callable,
    max_workers: int = 4,
    desc: str = "Processing"
) -> List[Any]:
    """
    并行处理任务列表
    
    Args:
        items: 待处理项目列表
        process_func: 处理函数
        max_workers: 最大工作线程数
        desc: 描述信息
        
    Returns:
        处理结果列表
    """
    logger.info(f"{desc}: {len(items)} items with {max_workers} workers")
    
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_func, item) for item in items]
        
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing item: {e}")
                results.append(None)
    
    return results