"""VLM代理模块"""
import logging
from typing import List, Dict, Tuple, Optional
import numpy as np
from PIL import Image

from src.models.vlm_wrapper import VLMWrapper

logger = logging.getLogger(__name__)


class VLMAgent:
    """VLM代理，用于验证和优化检测结果"""
    
    def __init__(
        self,
        vlm_model: VLMWrapper,
        verification_threshold: float = 0.3,
        batch_size: int = 32
    ):
        """
        初始化VLM Agent
        
        Args:
            vlm_model: VLM模型包装器
            verification_threshold: 验证阈值
            batch_size: 批处理大小
        """
        self.vlm_model = vlm_model
        self.verification_threshold = verification_threshold
        self.batch_size = batch_size
        
    def verify_detections(
        self,
        image_path: str,
        detections: Dict,
        query: Optional[str] = None
    ) -> Dict:
        """
        使用VLM验证检测结果
        
        Args:
            image_path: 图像路径
            detections: YOLO检测结果
            query: 可选的查询文本
            
        Returns:
            验证后的检测结果
        """
        if len(detections['boxes']) == 0:
            return detections
        
        # 加载图像
        image = Image.open(image_path)
        
        # 裁剪检测框区域
        cropped_images = self._crop_detections(image, detections['boxes'])
        
        # 构建验证查询
        if query:
            queries = [query] * len(cropped_images)
        else:
            queries = [f"a photo of a {name}" for name in detections['class_names']]
        
        # 批量验证
        verified_mask = []
        verification_scores = []
        
        for i in range(0, len(cropped_images), self.batch_size):
            batch_images = cropped_images[i:i + self.batch_size]
            batch_queries = queries[i:i + self.batch_size]
            
            batch_results = self.vlm_model.batch_verify(
                batch_images,
                batch_queries,
                self.verification_threshold
            )
            
            for is_match, score in batch_results:
                verified_mask.append(is_match)
                verification_scores.append(score)
        
        verified_mask = np.array(verified_mask)
        
        # 过滤未通过验证的检测
        filtered_detections = {
            'boxes': detections['boxes'][verified_mask],
            'scores': detections['scores'][verified_mask],
            'classes': detections['classes'][verified_mask],
            'class_names': [name for i, name in enumerate(detections['class_names']) if verified_mask[i]],
            'vlm_scores': np.array(verification_scores)[verified_mask]
        }
        
        logger.info(f"VLM verification: {verified_mask.sum()}/{len(verified_mask)} detections passed")
        
        return filtered_detections
    
    def re_classify(
        self,
        image_path: str,
        detections: Dict,
        class_names: List[str],
        top_k: int = 3
    ) -> Dict:
        """
        使用VLM重新分类检测结果
        
        Args:
            image_path: 图像路径
            detections: 检测结果
            class_names: 候选类别列表
            top_k: 返回top-k候选
            
        Returns:
            重新分类的结果
        """
        if len(detections['boxes']) == 0:
            return detections
        
        # 加载图像
        image = Image.open(image_path)
        
        # 裁剪检测框区域
        cropped_images = self._crop_detections(image, detections['boxes'])
        
        # 批量重新分类
        reclassified_results = []
        
        for cropped_img in cropped_images:
            top_classes = self.vlm_model.get_top_k_classes(
                cropped_img,
                class_names,
                k=top_k
            )
            reclassified_results.append(top_classes)
        
        # 更新检测结果
        updated_detections = detections.copy()
        updated_detections['vlm_predictions'] = reclassified_results
        
        return updated_detections
    
    def semantic_search(
        self,
        image_path: str,
        query: str,
        detections: Dict,
        top_k: int = 10
    ) -> Dict:
        """
        基于语义的检测搜索
        
        Args:
            image_path: 图像路径
            query: 查询文本
            detections: 检测结果
            top_k: 返回top-k结果
            
        Returns:
            按相似度排序的检测结果
        """
        if len(detections['boxes']) == 0:
            return detections
        
        # 加载图像
        image = Image.open(image_path)
        
        # 裁剪检测框区域
        cropped_images = self._crop_detections(image, detections['boxes'])
        
        # 计算与查询的相似度
        similarities = []
        for cropped_img in cropped_images:
            sim = self.vlm_model.compute_similarity(cropped_img, query)[0, 0]
            similarities.append(sim)
        
        similarities = np.array(similarities)
        
        # 排序并选择top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        sorted_detections = {
            'boxes': detections['boxes'][top_indices],
            'scores': detections['scores'][top_indices],
            'classes': detections['classes'][top_indices],
            'class_names': [detections['class_names'][i] for i in top_indices],
            'similarity_scores': similarities[top_indices]
        }
        
        return sorted_detections
    
    def _crop_detections(
        self,
        image: Image.Image,
        boxes: np.ndarray
    ) -> List[Image.Image]:
        """
        裁剪检测框区域
        
        Args:
            image: PIL图像
            boxes: 边界框数组 [N, 4] (x1, y1, x2, y2)
            
        Returns:
            裁剪后的图像列表
        """
        cropped_images = []
        
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            cropped = image.crop((x1, y1, x2, y2))
            cropped_images.append(cropped)
        
        return cropped_images
    
    def ensemble_confidence(
        self,
        detections: Dict,
        yolo_weight: float = 0.5,
        vlm_weight: float = 0.5
    ) -> Dict:
        """
        融合YOLO和VLM的置信度
        
        Args:
            detections: 包含YOLO分数和VLM分数的检测结果
            yolo_weight: YOLO权重
            vlm_weight: VLM权重
            
        Returns:
            更新置信度的检测结果
        """
        if 'vlm_scores' not in detections:
            return detections
        
        # 归一化权重
        total_weight = yolo_weight + vlm_weight
        yolo_weight /= total_weight
        vlm_weight /= total_weight
        
        # 融合分数
        ensemble_scores = (
            yolo_weight * detections['scores'] +
            vlm_weight * detections['vlm_scores']
        )
        
        detections['ensemble_scores'] = ensemble_scores
        detections['scores'] = ensemble_scores  # 更新主分数
        
        return detections