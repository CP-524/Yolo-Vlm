"""指标计算工具"""
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    计算两个边界框的IoU
    
    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
        
    Returns:
        IoU值
    """
    # 计算交集区域
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    if x2_inter < x1_inter or y2_inter < y1_inter:
        return 0.0
    
    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    
    # 计算并集区域
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    
    iou = inter_area / union_area if union_area > 0 else 0.0
    return iou


def match_predictions_to_ground_truth(
    pred_boxes: np.ndarray,
    pred_classes: np.ndarray,
    pred_scores: np.ndarray,
    gt_boxes: np.ndarray,
    gt_classes: np.ndarray,
    iou_threshold: float = 0.5
) -> Tuple[List[bool], List[int]]:
    """
    将预测框匹配到真实框
    
    Args:
        pred_boxes: 预测框 [N, 4]
        pred_classes: 预测类别 [N]
        pred_scores: 预测分数 [N]
        gt_boxes: 真实框 [M, 4]
        gt_classes: 真实类别 [M]
        iou_threshold: IoU阈值
        
    Returns:
        (是否为真阳性列表, 匹配的GT索引列表)
    """
    num_preds = len(pred_boxes)
    num_gts = len(gt_boxes)
    
    if num_preds == 0:
        return [], []
    
    if num_gts == 0:
        return [False] * num_preds, [-1] * num_preds
    
    # 按分数降序排序预测
    sorted_indices = np.argsort(pred_scores)[::-1]
    
    is_true_positive = [False] * num_preds
    matched_gt_idx = [-1] * num_preds
    gt_matched = [False] * num_gts
    
    # 贪心匹配
    for pred_idx in sorted_indices:
        pred_box = pred_boxes[pred_idx]
        pred_class = pred_classes[pred_idx]
        
        best_iou = 0.0
        best_gt_idx = -1
        
        # 找到最佳匹配的GT
        for gt_idx in range(num_gts):
            if gt_matched[gt_idx]:
                continue
            
            if gt_classes[gt_idx] != pred_class:
                continue
            
            iou = calculate_iou(pred_box, gt_boxes[gt_idx])
            
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        # 如果IoU超过阈值,标记为TP
        if best_iou >= iou_threshold and best_gt_idx >= 0:
            is_true_positive[pred_idx] = True
            matched_gt_idx[pred_idx] = best_gt_idx
            gt_matched[best_gt_idx] = True
    
    return is_true_positive, matched_gt_idx


def calculate_metrics(
    predictions: List[Dict],
    ground_truths: List[Dict],
    num_classes: int,
    iou_threshold: float = 0.5
) -> Dict:
    """
    计算检测指标
    
    Args:
        predictions: 预测结果列表
        ground_truths: 真实标注列表
        num_classes: 类别数量
        iou_threshold: IoU阈值
        
    Returns:
        指标字典
    """
    # 按类别收集预测和GT
    class_preds = defaultdict(list)  # {class_id: [(score, is_tp), ...]}
    class_gts = defaultdict(int)  # {class_id: count}
    
    for pred, gt in zip(predictions, ground_truths):
        pred_boxes = pred['boxes']
        pred_classes = pred['classes']
        pred_scores = pred['scores']
        
        gt_boxes = gt['boxes']
        gt_classes = gt['classes']
        
        # 统计GT数量
        for gt_class in gt_classes:
            class_gts[int(gt_class)] += 1
        
        # 匹配预测到GT
        is_tp, _ = match_predictions_to_ground_truth(
            pred_boxes, pred_classes, pred_scores,
            gt_boxes, gt_classes, iou_threshold
        )
        
        # 记录每个预测
        for i, (score, tp) in enumerate(zip(pred_scores, is_tp)):
            class_id = int(pred_classes[i])
            class_preds[class_id].append((float(score), tp))
    
    # 计算每个类别的AP
    aps = []
    precisions = []
    recalls = []
    f1_scores = []
    
    for class_id in range(num_classes):
        preds = class_preds[class_id]
        num_gt = class_gts[class_id]
        
        if num_gt == 0:
            aps.append(0.0)
            precisions.append(0.0)
            recalls.append(0.0)
            f1_scores.append(0.0)
            continue
        
        if len(preds) == 0:
            aps.append(0.0)
            precisions.append(0.0)
            recalls.append(0.0)
            f1_scores.append(0.0)
            continue
        
        # 按分数降序排序
        preds.sort(key=lambda x: x[0], reverse=True)
        
        # 计算precision-recall曲线
        tp_cumsum = 0
        fp_cumsum = 0
        precision_curve = []
        recall_curve = []
        
        for score, is_tp in preds:
            if is_tp:
                tp_cumsum += 1
            else:
                fp_cumsum += 1
            
            precision = tp_cumsum / (tp_cumsum + fp_cumsum)
            recall = tp_cumsum / num_gt
            
            precision_curve.append(precision)
            recall_curve.append(recall)
        
        # 计算AP (使用11点插值法)
        ap = 0.0
        for threshold in np.linspace(0, 1, 11):
            precisions_above_threshold = [
                p for p, r in zip(precision_curve, recall_curve) if r >= threshold
            ]
            if len(precisions_above_threshold) > 0:
                ap += max(precisions_above_threshold) / 11
        
        aps.append(ap)
        
        # 最终precision和recall
        final_precision = precision_curve[-1] if precision_curve else 0.0
        final_recall = recall_curve[-1] if recall_curve else 0.0
        final_f1 = 2 * final_precision * final_recall / (final_precision + final_recall) \
                   if (final_precision + final_recall) > 0 else 0.0
        
        precisions.append(final_precision)
        recalls.append(final_recall)
        f1_scores.append(final_f1)
    
    # 计算mAP
    map_score = np.mean(aps)
    
    return {
        'map': map_score,
        'ap_per_class': aps,
        'precision': precisions,
        'recall': recalls,
        'f1': f1_scores,
        'mean_precision': np.mean(precisions),
        'mean_recall': np.mean(recalls),
        'mean_f1': np.mean(f1_scores)
    }


def calculate_map(
    predictions: List[Dict],
    ground_truths: List[Dict],
    num_classes: int,
    iou_thresholds: List[float] = [0.5]
) -> Dict:
    """
    计算mAP (支持多个IoU阈值)
    
    Args:
        predictions: 预测结果列表
        ground_truths: 真实标注列表
        num_classes: 类别数量
        iou_thresholds: IoU阈值列表
        
    Returns:
        mAP字典
    """
    results = {}
    
    for iou_thresh in iou_thresholds:
        metrics = calculate_metrics(predictions, ground_truths, num_classes, iou_thresh)
        results[f'mAP@{iou_thresh:.2f}'] = metrics['map']
    
    # 计算mAP@[0.5:0.95]
    if len(iou_thresholds) > 1:
        results['mAP@[0.5:0.95]'] = np.mean([results[f'mAP@{t:.2f}'] for t in iou_thresholds])
    
    return results