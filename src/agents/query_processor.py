"""查询处理模块"""
import re
import logging
from typing import List, Dict, Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)


class QueryProcessor:
    """查询处理器，用于处理和解析检测查询"""
    
    def __init__(self, class_names: List[str]):
        """
        初始化查询处理器
        
        Args:
            class_names: 数据集类别名称列表
        """
        self.class_names = class_names
        self.class_name_to_id = {name: idx for idx, name in enumerate(class_names)}
        
    def parse_query(self, query: str) -> Dict[str, any]:
        """
        解析自然语言查询
        
        Args:
            query: 查询字符串，如 "find all planes"
            
        Returns:
            解析结果字典
        """
        query = query.lower().strip()
        
        # 提取目标类别
        target_classes = self._extract_classes(query)
        
        # 提取数量要求
        quantity_requirement = self._extract_quantity(query)
        
        # 提取位置要求
        location_requirement = self._extract_location(query)
        
        return {
            "original_query": query,
            "target_classes": target_classes,
            "quantity": quantity_requirement,
            "location": location_requirement
        }
    
    def _extract_classes(self, query: str) -> List[str]:
        """
        从查询中提取目标类别
        
        Args:
            query: 查询字符串
            
        Returns:
            匹配的类别名称列表
        """
        matched_classes = []
        
        for class_name in self.class_names:
            # 处理类别名称（如 baseball-diamond -> baseball diamond）
            class_variants = [
                class_name,
                class_name.replace('-', ' '),
                class_name.replace('-', '')
            ]
            
            # 检查是否匹配
            for variant in class_variants:
                if variant in query:
                    matched_classes.append(class_name)
                    break
        
        return matched_classes
    
    def _extract_quantity(self, query: str) -> Optional[Dict[str, any]]:
        """
        提取数量要求
        
        Args:
            query: 查询字符串
            
        Returns:
            数量要求字典或None
        """
        # 匹配 "all", "at least N", "exactly N", "more than N" 等
        patterns = [
            (r'all\s+(\w+)', 'all'),
            (r'at least (\d+)', 'at_least'),
            (r'exactly (\d+)', 'exactly'),
            (r'more than (\d+)', 'more_than'),
            (r'less than (\d+)', 'less_than')
        ]
        
        for pattern, requirement_type in patterns:
            match = re.search(pattern, query)
            if match:
                if requirement_type == 'all':
                    return {'type': 'all'}
                else:
                    count = int(match.group(1))
                    return {'type': requirement_type, 'count': count}
        
        return None
    
    def _extract_location(self, query: str) -> Optional[Dict[str, str]]:
        """
        提取位置要求
        
        Args:
            query: 查询字符串
            
        Returns:
            位置要求字典或None
        """
        location_keywords = {
            'top': ['top', 'upper'],
            'bottom': ['bottom', 'lower'],
            'left': ['left'],
            'right': ['right'],
            'center': ['center', 'middle']
        }
        
        for location, keywords in location_keywords.items():
            if any(kw in query for kw in keywords):
                return {'region': location}
        
        return None
    
    def filter_by_query(
        self, 
        detections: Dict,
        query_info: Dict
    ) -> Dict:
        """
        根据查询信息过滤检测结果
        
        Args:
            detections: 检测结果
            query_info: 解析的查询信息
            
        Returns:
            过滤后的检测结果
        """
        # 按类别过滤
        target_classes = query_info.get('target_classes', [])
        if target_classes:
            target_ids = [self.class_name_to_id[name] for name in target_classes]
            mask = np.isin(detections['classes'], target_ids)
            
            detections = {
                'boxes': detections['boxes'][mask],
                'scores': detections['scores'][mask],
                'classes': detections['classes'][mask],
                'class_names': [name for i, name in enumerate(detections['class_names']) if mask[i]]
            }
        
        # 按数量过滤
        quantity = query_info.get('quantity')
        if quantity and quantity['type'] != 'all':
            detections = self._filter_by_quantity(detections, quantity)
        
        # 按位置过滤
        location = query_info.get('location')
        if location:
            detections = self._filter_by_location(detections, location)
        
        return detections
    
    def _filter_by_quantity(self, detections: Dict, quantity: Dict) -> Dict:
        """按数量要求过滤"""
        num_detections = len(detections['boxes'])
        required_count = quantity.get('count', 0)
        requirement_type = quantity['type']
        
        # 根据要求类型过滤
        if requirement_type == 'exactly' and num_detections != required_count:
            return self._empty_detections()
        elif requirement_type == 'at_least' and num_detections < required_count:
            return self._empty_detections()
        elif requirement_type == 'more_than' and num_detections <= required_count:
            return self._empty_detections()
        elif requirement_type == 'less_than' and num_detections >= required_count:
            # 只保留前N-1个最高置信度的检测
            if num_detections > 0:
                top_k = min(required_count - 1, num_detections)
                top_indices = np.argsort(detections['scores'])[::-1][:top_k]
                detections = self._select_detections(detections, top_indices)
        
        return detections
    
    def _filter_by_location(self, detections: Dict, location: Dict) -> Dict:
        """按位置要求过滤"""
        # 简化实现：基于边界框中心点
        if len(detections['boxes']) == 0:
            return detections
        
        boxes = detections['boxes']
        centers = (boxes[:, :2] + boxes[:, 2:]) / 2  # 计算中心点
        
        # 根据区域过滤（简化实现）
        region = location['region']
        # 这里需要知道图像尺寸，暂时返回原检测
        logger.warning("Location-based filtering not fully implemented")
        
        return detections
    
    def _select_detections(self, detections: Dict, indices: np.ndarray) -> Dict:
        """选择指定索引的检测"""
        return {
            'boxes': detections['boxes'][indices],
            'scores': detections['scores'][indices],
            'classes': detections['classes'][indices],
            'class_names': [detections['class_names'][i] for i in indices]
        }
    
    def _empty_detections(self) -> Dict:
        """返回空检测结果"""
        return {
            'boxes': np.array([]),
            'scores': np.array([]),
            'classes': np.array([]),
            'class_names': []
        }