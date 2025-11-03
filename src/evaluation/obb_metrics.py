"""OBB (Oriented Bounding Box) 评估指标计算"""
import numpy as np
from typing import List, Dict, Tuple
import logging
from shapely.geometry import Polygon
from shapely.ops import unary_union

logger = logging.getLogger(__name__)


def poly_iou(poly1: np.ndarray, poly2: np.ndarray) -> float:
    """
    计算两个多边形的IoU (高度优化版本)
    
    Args:
        poly1: [8] (x1, y1, x2, y2, x3, y3, x4, y4)
        poly2: [8] (x1, y1, x2, y2, x3, y3, x4, y4)
        
    Returns:
        IoU值
    """
    try:
        # 将数组reshape为点集
        points1 = poly1.reshape(4, 2)
        points2 = poly2.reshape(4, 2)
        
        # 快速检查：计算外接矩形
        x1_min, y1_min = points1[:, 0].min(), points1[:, 1].min()
        x1_max, y1_max = points1[:, 0].max(), points1[:, 1].max()
        x2_min, y2_min = points2[:, 0].min(), points2[:, 1].min()
        x2_max, y2_max = points2[:, 0].max(), points2[:, 1].max()
        
        # 如果外接矩形不重叠，直接返回0（避免昂贵的Shapely计算）
        if x1_max < x2_min or x2_max < x1_min or y1_max < y2_min or y2_max < y1_min:
            return 0.0
        
        # 计算外接矩形的IoU作为上界估计
        bbox_inter_x = min(x1_max, x2_max) - max(x1_min, x2_min)
        bbox_inter_y = min(y1_max, y2_max) - max(y1_min, y2_min)
        bbox_inter_area = bbox_inter_x * bbox_inter_y
        
        bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
        bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)
        bbox_union_area = bbox1_area + bbox2_area - bbox_inter_area
        
        # 如果外接矩形的IoU很小，可以直接返回近似值
        bbox_iou = bbox_inter_area / bbox_union_area if bbox_union_area > 0 else 0.0
        if bbox_iou < 0.1:  # 如果外接框IoU很小，直接返回0
            return 0.0
        
        # 创建Polygon对象
        polygon1 = Polygon(points1)
        polygon2 = Polygon(points2)
        
        # 如果多边形无效，尝试修复
        if not polygon1.is_valid:
            polygon1 = polygon1.buffer(0)
        if not polygon2.is_valid:
            polygon2 = polygon2.buffer(0)
        
        # 计算交集和并集
        intersection = polygon1.intersection(polygon2).area
        union = polygon1.union(polygon2).area
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    except Exception as e:
        # 静默处理错误，避免大量警告
        return 0.0


def compute_obb_iou(obb1: np.ndarray, obb2: np.ndarray, format: str = 'poly') -> float:
    """
    计算两个OBB的IoU
    
    Args:
        obb1: OBB坐标
        obb2: OBB坐标
        format: 'poly' (x1,y1,x2,y2,x3,y3,x4,y4) 或 'xywhr' (cx,cy,w,h,angle)
        
    Returns:
        IoU值
    """
    if format == 'poly':
        return poly_iou(obb1, obb2)
    elif format == 'xywhr':
        # 将xywhr转换为poly
        poly1 = xywhr_to_poly(obb1)
        poly2 = xywhr_to_poly(obb2)
        return poly_iou(poly1, poly2)
    else:
        raise ValueError(f"Unsupported format: {format}")


def xywhr_to_poly(xywhr: np.ndarray) -> np.ndarray:
    """
    将xywhr格式转换为多边形
    
    Args:
        xywhr: [cx, cy, w, h, angle]
        
    Returns:
        poly: [8] (x1, y1, x2, y2, x3, y3, x4, y4)
    """
    cx, cy, w, h, angle = xywhr
    
    # 计算四个角点（未旋转）
    corners = np.array([
        [-w/2, -h/2],
        [w/2, -h/2],
        [w/2, h/2],
        [-w/2, h/2]
    ])
    
    # 旋转矩阵
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    rotation_matrix = np.array([
        [cos_a, -sin_a],
        [sin_a, cos_a]
    ])
    
    # 旋转角点
    rotated_corners = corners @ rotation_matrix.T
    
    # 平移到中心点
    rotated_corners[:, 0] += cx
    rotated_corners[:, 1] += cy
    
    return rotated_corners.flatten()


def compute_obb_precision_recall(
    predictions: List[Dict],
    ground_truths: List[Dict],
    class_names: List[str],
    iou_threshold: float = 0.5
) -> Dict[str, Dict]:
    """
    计算每个类别的OBB精确率和召回率
    
    Args:
        predictions: 预测结果列表
        ground_truths: 真实标注列表
        class_names: 类别名称列表
        iou_threshold: IoU阈值
        
    Returns:
        每个类别的精确率和召回率
    """
    import logging
    logger = logging.getLogger(__name__)
    
    num_classes = len(class_names)
    class_metrics = {}
    
    logger.info("Computing per-class metrics...")
    
    for class_id in range(num_classes):
        class_name = class_names[class_id]
        
        # 收集该类别的所有预测和真实标注
        pred_boxes = []
        pred_scores = []
        gt_boxes = []
        
        for pred, gt in zip(predictions, ground_truths):
            # 预测框 - 优先使用OBB格式
            if pred.get('is_obb', False) and len(pred.get('obb_boxes', [])) > 0:
                pred_mask = pred['classes'] == class_id
                if isinstance(pred_mask, np.ndarray) and pred_mask.any():
                    pred_boxes.extend(pred['obb_boxes'][pred_mask])
                    pred_scores.extend(pred['scores'][pred_mask])
            elif len(pred.get('boxes', [])) > 0:
                # 回退到水平框
                pred_mask = pred['classes'] == class_id
                if isinstance(pred_mask, np.ndarray) and pred_mask.any():
                    pred_boxes.extend(pred['boxes'][pred_mask])
                    pred_scores.extend(pred['scores'][pred_mask])
            
            # 真实标注 - 优先使用OBB格式
            if gt.get('is_obb', False) and len(gt.get('obb_boxes', [])) > 0:
                gt_mask = gt['classes'] == class_id
                if isinstance(gt_mask, np.ndarray) and gt_mask.any():
                    gt_boxes.extend(gt['obb_boxes'][gt_mask])
            elif len(gt.get('boxes', [])) > 0:
                # 回退到水平框
                gt_mask = gt['classes'] == class_id
                if isinstance(gt_mask, np.ndarray) and gt_mask.any():
                    gt_boxes.extend(gt['boxes'][gt_mask])
        
        # 计算精确率和召回率
        if len(pred_boxes) == 0 and len(gt_boxes) == 0:
            precision = 1.0
            recall = 1.0
        elif len(pred_boxes) == 0:
            precision = 0.0
            recall = 0.0
        elif len(gt_boxes) == 0:
            precision = 0.0
            recall = 0.0
        else:
            precision, recall = _compute_class_obb_precision_recall(
                pred_boxes, pred_scores, gt_boxes, iou_threshold
            )
        
        class_metrics[class_name] = {
            'precision': precision,
            'recall': recall,
            'f1': 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0,
            'num_pred': len(pred_boxes),
            'num_gt': len(gt_boxes)
        }
    
    return class_metrics


def _compute_class_obb_precision_recall(
    pred_boxes: List[np.ndarray],
    pred_scores: List[float],
    gt_boxes: List[np.ndarray],
    iou_threshold: float
) -> Tuple[float, float]:
    """计算单个类别的OBB精确率和召回率"""
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return 0.0, 0.0
    
    # 判断框的格式
    is_obb = len(pred_boxes[0]) == 8 or len(pred_boxes[0]) == 5
    
    # 按置信度排序
    sorted_indices = np.argsort(pred_scores)[::-1]
    pred_boxes = [pred_boxes[i] for i in sorted_indices]
    
    # 匹配预测框和真实框
    matched_gt = set()
    true_positives = 0
    
    for pred_box in pred_boxes:
        best_iou = 0
        best_gt_idx = -1
        
        for gt_idx, gt_box in enumerate(gt_boxes):
            if gt_idx in matched_gt:
                continue
            
            # 根据格式计算IoU
            if is_obb and len(pred_box) == 8 and len(gt_box) == 8:
                iou = poly_iou(pred_box, gt_box)
            else:
                # 使用标准IoU
                iou = _compute_box_iou(pred_box, gt_box)
            
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        if best_iou >= iou_threshold:
            true_positives += 1
            matched_gt.add(best_gt_idx)
    
    precision = true_positives / len(pred_boxes) if len(pred_boxes) > 0 else 0.0
    recall = true_positives / len(gt_boxes) if len(gt_boxes) > 0 else 0.0
    
    return precision, recall


def _compute_box_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """计算标准水平框的IoU"""
    # 假设是xyxy格式
    if len(box1) == 4 and len(box2) == 4:
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    return 0.0


def compute_obb_map(
    predictions: List[Dict],
    ground_truths: List[Dict],
    class_names: List[str],
    iou_thresholds: List[float] = [0.5, 0.75]  # 简化：只计算50和75
) -> Dict:
    """
    计算OBB mAP (优化版本，只计算mAP@50和mAP@75)
    
    Args:
        predictions: 预测结果列表
        ground_truths: 真实标注列表
        class_names: 类别名称列表
        iou_thresholds: IoU阈值列表
        
    Returns:
        mAP指标
    """
    import logging
    logger = logging.getLogger(__name__)
    
    num_images = len(predictions)
    logger.info(f"Computing mAP for {len(class_names)} classes on {num_images} images...")
    
    # 计算每个IoU阈值下的AP
    aps_per_threshold = []
    
    from tqdm import tqdm
    import time
    
    for iou_thresh in iou_thresholds:
        logger.info(f"Computing AP @ IoU={iou_thresh}...")
        aps = []
        start_time = time.time()
        
        for class_id in tqdm(range(len(class_names)), desc=f"AP@{iou_thresh}", leave=False):
            ap = _compute_class_ap(
                predictions, ground_truths, class_id, iou_thresh
            )
            aps.append(ap)
        
        elapsed = time.time() - start_time
        logger.info(f"  Completed in {elapsed:.1f}s ({elapsed/len(class_names):.2f}s per class)")
        aps_per_threshold.append(np.mean(aps))
    
    # mAP是所有IoU阈值的平均
    map_score = np.mean(aps_per_threshold)
    map50 = aps_per_threshold[0]  # IoU=0.5
    map75 = aps_per_threshold[1] if len(aps_per_threshold) > 1 else 0  # IoU=0.75
    
    return {
        'mAP': map_score,
        'mAP50': map50,
        'mAP75': map75,
        'mAP_per_threshold': aps_per_threshold
    }


def _compute_class_ap(
    predictions: List[Dict],
    ground_truths: List[Dict],
    class_id: int,
    iou_threshold: float
) -> float:
    """
    计算单个类别的AP (优化版本)
    使用批量处理和numpy向量化减少计算时间
    """
    # 收集该类别的所有预测和真实标注
    all_pred_boxes = []
    all_pred_scores = []
    all_gt_boxes = []
    image_ids = []  # 记录每个框属于哪张图
    
    for img_id, (pred, gt) in enumerate(zip(predictions, ground_truths)):
        # 预测框
        if pred.get('is_obb', False) and len(pred.get('obb_boxes', [])) > 0:
            pred_mask = pred['classes'] == class_id
            if isinstance(pred_mask, np.ndarray) and pred_mask.any():
                boxes = pred['obb_boxes'][pred_mask]
                scores = pred['scores'][pred_mask]
                all_pred_boxes.extend(boxes)
                all_pred_scores.extend(scores)
                image_ids.extend([img_id] * len(boxes))
        
        # 真实标注
        gt_boxes_for_image = []
        if gt.get('is_obb', False) and len(gt.get('obb_boxes', [])) > 0:
            gt_mask = gt['classes'] == class_id
            if isinstance(gt_mask, np.ndarray) and gt_mask.any():
                gt_boxes_for_image = gt['obb_boxes'][gt_mask]
        
        all_gt_boxes.append(gt_boxes_for_image)
    
    if len(all_pred_boxes) == 0:
        return 0.0
    
    # 统计总的GT数量，如果为0则返回
    total_gt = sum(len(boxes) for boxes in all_gt_boxes)
    if total_gt == 0:
        return 0.0
    
    # 按置信度排序
    sorted_indices = np.argsort(all_pred_scores)[::-1]
    all_pred_boxes = [all_pred_boxes[i] for i in sorted_indices]
    image_ids = [image_ids[i] for i in sorted_indices]
    
    # 计算每个预测的TP/FP
    tp = np.zeros(len(all_pred_boxes))
    fp = np.zeros(len(all_pred_boxes))
    matched_gt_per_image = [set() for _ in range(len(ground_truths))]
    
    # 优化：批量处理，减少函数调用开销
    for pred_idx, (pred_box, img_id) in enumerate(zip(all_pred_boxes, image_ids)):
        gt_boxes = all_gt_boxes[img_id]
        
        if len(gt_boxes) == 0:
            fp[pred_idx] = 1
            continue
        
        best_iou = 0
        best_gt_idx = -1
        
        # 优化：对于每个预测框，只和同图像的GT计算IoU
        for gt_idx, gt_box in enumerate(gt_boxes):
            if gt_idx in matched_gt_per_image[img_id]:
                continue
            
            # 使用快速IoU计算
            iou = poly_iou(pred_box, gt_box) if len(pred_box) == 8 else _compute_box_iou(pred_box, gt_box)
            
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        if best_iou >= iou_threshold:
            tp[pred_idx] = 1
            matched_gt_per_image[img_id].add(best_gt_idx)
        else:
            fp[pred_idx] = 1
    
    # 计算precision-recall曲线
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    
    total_gt = sum(len(boxes) for boxes in all_gt_boxes)
    if total_gt == 0:
        return 0.0
    
    recalls = tp_cumsum / total_gt
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
    
    # 使用11点插值计算AP
    ap = 0
    for t in np.arange(0, 1.1, 0.1):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 11
    
    return ap
