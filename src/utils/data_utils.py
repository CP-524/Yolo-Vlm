"""数据预处理和加载工具"""
import json
import yaml
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def load_annotations(annotation_path: str, format: str = 'yolo') -> Dict:
    """
    加载标注文件
    
    Args:
        annotation_path: 标注文件路径
        format: 格式 ('yolo', 'coco', 'voc')
        
    Returns:
        标注字典
    """
    annotation_path = Path(annotation_path)
    
    if format == 'yolo':
        return load_yolo_annotations(annotation_path)
    elif format == 'coco':
        return load_coco_annotations(annotation_path)
    else:
        raise ValueError(f"Unsupported format: {format}")


def load_yolo_annotations(txt_path: Path) -> Dict:
    """
    加载YOLO格式标注
    支持两种格式:
    1. 标准YOLO: class_id x_center y_center width height (5个数字)
    2. DOTA旋转框: class_id x1 y1 x2 y2 x3 y3 x4 y4 (9个数字)
    
    Args:
        txt_path: 标注文件路径
        
    Returns:
        标注字典
    """
    if not txt_path.exists():
        return {
            'boxes': np.array([]),
            'classes': np.array([]),
            'format': 'yolo'
        }
    
    annotations = []
    obb_annotations = []
    is_obb = False
    
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            
            if len(parts) >= 9:
                # DOTA旋转框格式: class_id x1 y1 x2 y2 x3 y3 x4 y4
                is_obb = True
                class_id = int(parts[0])
                points = list(map(float, parts[1:9]))
                
                # 保存原始旋转框坐标
                obb_annotations.append([class_id] + points)
                
                # 同时计算水平边界框用于兼容
                xs = [points[0], points[2], points[4], points[6]]
                ys = [points[1], points[3], points[5], points[7]]
                
                x_min = min(xs)
                x_max = max(xs)
                y_min = min(ys)
                y_max = max(ys)
                
                # 转换为中心点格式
                x_center = (x_min + x_max) / 2
                y_center = (y_min + y_max) / 2
                width = x_max - x_min
                height = y_max - y_min
                
                annotations.append([class_id, x_center, y_center, width, height])
                
            elif len(parts) >= 5:
                # 标准YOLO格式: class_id x_center y_center width height
                class_id = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:5])
                annotations.append([class_id, x_center, y_center, width, height])
    
    if not annotations:
        return {
            'boxes': np.array([]),
            'classes': np.array([]),
            'obb_boxes': np.array([]),
            'format': 'yolo',
            'is_obb': False
        }
    
    annotations = np.array(annotations)
    result = {
        'boxes': annotations[:, 1:5],  # [x_center, y_center, w, h]
        'classes': annotations[:, 0].astype(int),
        'format': 'yolo',
        'is_obb': is_obb
    }
    
    if is_obb:
        obb_annotations = np.array(obb_annotations)
        result['obb_boxes'] = obb_annotations[:, 1:9]  # [x1, y1, x2, y2, x3, y3, x4, y4]
    else:
        result['obb_boxes'] = np.array([])
    
    return result


def load_coco_annotations(json_path: Path) -> Dict:
    """加载COCO格式标注"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # 简化实现
    boxes = []
    classes = []
    
    for ann in data.get('annotations', []):
        bbox = ann['bbox']  # [x, y, width, height]
        category_id = ann['category_id']
        
        boxes.append(bbox)
        classes.append(category_id)
    
    return {
        'boxes': np.array(boxes),
        'classes': np.array(classes),
        'format': 'coco'
    }


def yolo_to_xyxy(boxes: np.ndarray, img_width: int, img_height: int) -> np.ndarray:
    """
    将YOLO格式(归一化的中心点坐标)转换为xyxy格式
    
    Args:
        boxes: [N, 4] (x_center, y_center, w, h) 归一化
        img_width: 图像宽度
        img_height: 图像高度
        
    Returns:
        [N, 4] (x1, y1, x2, y2) 绝对坐标
    """
    if len(boxes) == 0:
        return np.array([])
    
    boxes_xyxy = boxes.copy()
    
    # 反归一化
    boxes_xyxy[:, [0, 2]] *= img_width
    boxes_xyxy[:, [1, 3]] *= img_height
    
    # 转换为xyxy
    x_center, y_center, w, h = boxes_xyxy[:, 0], boxes_xyxy[:, 1], boxes_xyxy[:, 2], boxes_xyxy[:, 3]
    x1 = x_center - w / 2
    y1 = y_center - h / 2
    x2 = x_center + w / 2
    y2 = y_center + h / 2
    
    return np.stack([x1, y1, x2, y2], axis=1)


def xyxy_to_yolo(boxes: np.ndarray, img_width: int, img_height: int) -> np.ndarray:
    """
    将xyxy格式转换为YOLO格式(归一化)
    
    Args:
        boxes: [N, 4] (x1, y1, x2, y2) 绝对坐标
        img_width: 图像宽度
        img_height: 图像高度
        
    Returns:
        [N, 4] (x_center, y_center, w, h) 归一化
    """
    if len(boxes) == 0:
        return np.array([])
    
    boxes_yolo = boxes.copy()
    
    # 计算中心点和宽高
    x1, y1, x2, y2 = boxes_yolo[:, 0], boxes_yolo[:, 1], boxes_yolo[:, 2], boxes_yolo[:, 3]
    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    
    # 归一化
    x_center /= img_width
    y_center /= img_height
    w /= img_width
    h /= img_height
    
    return np.stack([x_center, y_center, w, h], axis=1)


def save_predictions(
    predictions: Dict,
    output_path: str,
    format: str = 'yolo',
    img_width: Optional[int] = None,
    img_height: Optional[int] = None
):
    """
    保存预测结果
    
    Args:
        predictions: 预测字典
        output_path: 输出路径
        format: 格式
        img_width: 图像宽度(YOLO格式需要)
        img_height: 图像高度(YOLO格式需要)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'yolo':
        if img_width is None or img_height is None:
            raise ValueError("YOLO format requires img_width and img_height")
        
        save_yolo_predictions(predictions, output_path, img_width, img_height)
    elif format == 'json':
        save_json_predictions(predictions, output_path)
    else:
        raise ValueError(f"Unsupported format: {format}")


def save_yolo_predictions(
    predictions: Dict,
    output_path: Path,
    img_width: int,
    img_height: int
):
    """保存YOLO格式预测"""
    boxes = predictions['boxes']
    classes = predictions['classes']
    scores = predictions.get('scores', None)
    
    if len(boxes) == 0:
        # 创建空文件
        output_path.write_text("")
        return
    
    # 转换为YOLO格式
    boxes_yolo = xyxy_to_yolo(boxes, img_width, img_height)
    
    with open(output_path, 'w') as f:
        for i, (box, cls) in enumerate(zip(boxes_yolo, classes)):
            line = f"{int(cls)} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}"
            if scores is not None:
                line += f" {scores[i]:.6f}"
            f.write(line + "\n")


def save_json_predictions(predictions: Dict, output_path: Path):
    """保存JSON格式预测"""
    # 转换numpy数组为列表
    output_data = {}
    for key, value in predictions.items():
        if isinstance(value, np.ndarray):
            output_data[key] = value.tolist()
        else:
            output_data[key] = value
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)


def load_yaml_config(config_path: str) -> Dict:
    """加载YAML配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def get_image_list(image_dir: str, extensions: List[str] = ['.jpg', '.jpeg', '.png']) -> List[Path]:
    """
    获取目录下的所有图像文件
    
    Args:
        image_dir: 图像目录
        extensions: 支持的扩展名
        
    Returns:
        图像路径列表
    """
    image_dir = Path(image_dir)
    image_files = []
    
    for ext in extensions:
        # 只搜索小写扩展名,避免在Windows上重复
        image_files.extend(image_dir.glob(f'*{ext}'))
    
    # 去重(以防万一)并排序
    return sorted(list(set(image_files)))


def obb_to_poly(obb: np.ndarray, img_width: int, img_height: int) -> np.ndarray:
    """
    将归一化的OBB坐标转换为绝对坐标的多边形
    
    Args:
        obb: [8] (x1, y1, x2, y2, x3, y3, x4, y4) 归一化坐标
        img_width: 图像宽度
        img_height: 图像高度
        
    Returns:
        [8] 绝对坐标的多边形点(扁平数组)
    """
    poly = obb.copy()
    poly[0::2] *= img_width   # x坐标
    poly[1::2] *= img_height  # y坐标
    return poly


def poly_to_obb(poly: np.ndarray, img_width: int, img_height: int) -> np.ndarray:
    """
    将绝对坐标的多边形转换为归一化的OBB坐标
    
    Args:
        poly: [4, 2] 或 [8] 绝对坐标的多边形点
        img_width: 图像宽度
        img_height: 图像高度
        
    Returns:
        [8] 归一化的OBB坐标
    """
    if poly.shape == (4, 2):
        poly = poly.flatten()
    
    obb = poly.copy()
    obb[0::2] /= img_width  # x坐标
    obb[1::2] /= img_height  # y坐标
    return obb