"""检测Agent（文章中的Pipeline）"""
import logging
from typing import Dict, Optional, List
from pathlib import Path

from src.models.yolo_wrapper import YOLOWrapper
from src.agents.vlm_agent import VLMAgent
from src.agents.query_processor import QueryProcessor

logger = logging.getLogger(__name__)


class DetectionAgent:
    """
    检测Agent - 整合YOLO、VLM和查询处理的完整pipeline
    实现文章中描述的Agentic方法
    """
    
    def __init__(
        self,
        yolo_model: YOLOWrapper,
        vlm_agent: Optional[VLMAgent] = None,
        query_processor: Optional[QueryProcessor] = None,
        enable_vlm: bool = True,
        enable_query_processing: bool = True
    ):
        """
        初始化检测Agent
        
        Args:
            yolo_model: YOLO模型包装器
            vlm_agent: VLM代理(可选)
            query_processor: 查询处理器(可选)
            enable_vlm: 是否启用VLM验证
            enable_query_processing: 是否启用查询处理
        """
        self.yolo_model = yolo_model
        self.vlm_agent = vlm_agent
        self.query_processor = query_processor
        self.enable_vlm = enable_vlm and vlm_agent is not None
        self.enable_query_processing = enable_query_processing and query_processor is not None
        
        logger.info(f"DetectionAgent initialized - VLM: {self.enable_vlm}, QueryProc: {self.enable_query_processing}")
    
    def detect(
        self,
        image_path: str,
        query: Optional[str] = None,
        conf_threshold: Optional[float] = None,
        enable_vlm_verification: Optional[bool] = None
    ) -> Dict:
        """
        执行检测pipeline
        
        Args:
            image_path: 图像路径
            query: 可选的自然语言查询
            conf_threshold: 置信度阈值(可选)
            enable_vlm_verification: 是否启用VLM验证(覆盖默认设置)
            
        Returns:
            检测结果字典
        """
        logger.info(f"Processing image: {image_path}")
        
        # Step 1: YOLO检测
        detections = self.yolo_model.predict(
            image_path,
            conf=conf_threshold
        )
        logger.info(f"YOLO detected {len(detections['boxes'])} objects")
        
        # Step 2: 查询处理和过滤
        if self.enable_query_processing and query:
            query_info = self.query_processor.parse_query(query)
            detections = self.query_processor.filter_by_query(detections, query_info)
            logger.info(f"After query filtering: {len(detections['boxes'])} objects")
        
        # Step 3: VLM验证
        use_vlm = self.enable_vlm if enable_vlm_verification is None else enable_vlm_verification
        if use_vlm and len(detections['boxes']) > 0:
            detections = self.vlm_agent.verify_detections(
                image_path,
                detections,
                query=query
            )
            logger.info(f"After VLM verification: {len(detections['boxes'])} objects")
        
        return detections
    
    def detect_batch(
        self,
        image_paths: List[str],
        queries: Optional[List[str]] = None,
        conf_threshold: Optional[float] = None
    ) -> List[Dict]:
        """
        批量检测
        
        Args:
            image_paths: 图像路径列表
            queries: 查询列表(可选)
            conf_threshold: 置信度阈值
            
        Returns:
            检测结果列表
        """
        results = []
        
        if queries is None:
            queries = [None] * len(image_paths)
        
        for img_path, query in zip(image_paths, queries):
            result = self.detect(img_path, query, conf_threshold)
            results.append(result)
        
        return results
    
    def baseline_detect(
        self,
        image_path: str,
        conf_threshold: Optional[float] = None
    ) -> Dict:
        """
        基线检测(仅YOLO,不使用Agent功能)
        
        Args:
            image_path: 图像路径
            conf_threshold: 置信度阈值
            
        Returns:
            检测结果
        """
        return self.yolo_model.predict(image_path, conf=conf_threshold)
    
    def compare_methods(
        self,
        image_path: str,
        query: Optional[str] = None,
        conf_threshold: Optional[float] = None
    ) -> Dict[str, Dict]:
        """
        比较基线方法和Agentic方法
        
        Args:
            image_path: 图像路径
            query: 查询
            conf_threshold: 置信度阈值
            
        Returns:
            包含两种方法结果的字典
        """
        # 基线方法
        baseline_results = self.baseline_detect(image_path, conf_threshold)
        
        # Agentic方法
        agentic_results = self.detect(image_path, query, conf_threshold)
        
        return {
            'baseline': baseline_results,
            'agentic': agentic_results
        }
    
    def semantic_search(
        self,
        image_path: str,
        query: str,
        top_k: int = 10,
        conf_threshold: Optional[float] = None
    ) -> Dict:
        """
        基于语义的目标搜索
        
        Args:
            image_path: 图像路径
            query: 查询文本
            top_k: 返回top-k结果
            conf_threshold: 置信度阈值
            
        Returns:
            按相似度排序的检测结果
        """
        if not self.enable_vlm:
            logger.warning("VLM is disabled, falling back to baseline detection")
            return self.baseline_detect(image_path, conf_threshold)
        
        # YOLO检测
        detections = self.yolo_model.predict(image_path, conf=conf_threshold)
        
        # VLM语义搜索
        results = self.vlm_agent.semantic_search(
            image_path,
            query,
            detections,
            top_k=top_k
        )
        
        return results
    
    def get_pipeline_info(self) -> Dict:
        """
        获取pipeline配置信息
        
        Returns:
            配置信息字典
        """
        return {
            'yolo_model': self.yolo_model.model_name if hasattr(self.yolo_model, 'model_name') else 'unknown',
            'vlm_enabled': self.enable_vlm,
            'query_processing_enabled': self.enable_query_processing,
            'vlm_model': self.vlm_agent.vlm_model.model_name if self.enable_vlm else None
        }