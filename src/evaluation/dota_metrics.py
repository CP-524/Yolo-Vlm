"""DOTA数据集评估指标计算"""
import numpy as np
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    计算两个边界框的IoU
    
    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
        
    Returns:
        IoU值
    """
    # 计算交集
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    
    # 计算并集
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def compute_precision_recall(
    predictions: List[Dict],
    ground_truths: List[Dict],
    class_names: List[str],
    iou_threshold: float = 0.5
) -> Dict[str, Dict]:
    """
    计算每个类别的精确率和召回率
    
    Args:
        predictions: 预测结果列表
        ground_truths: 真实标注列表
        class_names: 类别名称列表
        iou_threshold: IoU阈值
        
    Returns:
        每个类别的精确率和召回率
    """
    num_classes = len(class_names)
    class_metrics = {}
    
    for class_id in range(num_classes):
        class_name = class_names[class_id]
        
        # 收集该类别的所有预测和真实标注
        pred_boxes = []
        pred_scores = []
        gt_boxes = []
        
        for pred, gt in zip(predictions, ground_truths):
            # 预测框
            if len(pred['boxes']) > 0:
                pred_mask = pred['classes'] == class_id
                if isinstance(pred_mask, np.ndarray) and pred_mask.any():
                    pred_boxes.extend(pred['boxes'][pred_mask])
                    pred_scores.extend(pred['scores'][pred_mask])
                elif isinstance(pred_mask, bool) and pred_mask:
                    pred_boxes.extend(pred['boxes'])
                    pred_scores.extend(pred['scores'])
            
            # 真实标注
            if len(gt['boxes']) > 0:
                gt_mask = gt['classes'] == class_id
                if isinstance(gt_mask, np.ndarray) and gt_mask.any():
                    gt_boxes.extend(gt['boxes'][gt_mask])
                elif isinstance(gt_mask, bool) and gt_mask:
                    gt_boxes.extend(gt['boxes'])
        
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
            precision, recall = _compute_class_precision_recall(
                pred_boxes, pred_scores, gt_boxes, iou_threshold
            )
        
        class_metrics[class_name] = {
            'precision': precision,
            'recall': recall,
            'f1': 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        }
    
    return class_metrics


def _compute_class_precision_recall(
    pred_boxes: List[np.ndarray],
    pred_scores: List[float],
    gt_boxes: List[np.ndarray],
    iou_threshold: float
) -> Tuple[float, float]:
    """计算单个类别的精确率和召回率"""
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return 0.0, 0.0
    
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
            
            iou = compute_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        if best_iou >= iou_threshold:
            true_positives += 1
            matched_gt.add(best_gt_idx)
    
    precision = true_positives / len(pred_boxes) if len(pred_boxes) > 0 else 0.0
    recall = true_positives / len(gt_boxes) if len(gt_boxes) > 0 else 0.0
    
    return precision, recall


def compute_dota_metrics(
    predictions: List[Dict],
    ground_truths: List[Dict],
    class_names: List[str],
    iou_threshold: float = 0.5
) -> Dict:
    """
    计算DOTA数据集的评估指标
    
    Args:
        predictions: 预测结果列表
        ground_truths: 真实标注列表
        class_names: 类别名称列表
        iou_threshold: IoU阈值
        
    Returns:
        评估指标字典
    """
    logger.info(f"Computing DOTA metrics for {len(predictions)} images")
    
    # 计算每个类别的精确率和召回率
    class_metrics = compute_precision_recall(
        predictions, ground_truths, class_names, iou_threshold
    )
    
    # 计算平均指标
    precisions = [metrics['precision'] for metrics in class_metrics.values()]
    recalls = [metrics['recall'] for metrics in class_metrics.values()]
    f1_scores = [metrics['f1'] for metrics in class_metrics.values()]
    
    # 计算mAP (简化版本)
    mean_precision = np.mean(precisions)
    mean_recall = np.mean(recalls)
    mean_f1 = np.mean(f1_scores)
    
    # 计算mAP (使用精确率-召回率曲线下面积)
    map_score = _compute_map(predictions, ground_truths, class_names, iou_threshold)
    
    return {
        'map': map_score,
        'mean_precision': mean_precision,
        'mean_recall': mean_recall,
        'mean_f1': mean_f1,
        'precision': precisions,
        'recall': recalls,
        'f1': f1_scores,
        'class_metrics': class_metrics
    }


def _compute_map(
    predictions: List[Dict],
    ground_truths: List[Dict],
    class_names: List[str],
    iou_threshold: float
) -> float:
    """计算mAP (简化版本)"""
    num_classes = len(class_names)
    aps = []
    
    for class_id in range(num_classes):
        # 收集该类别的所有预测和真实标注
        pred_scores = []
        pred_boxes = []
        gt_boxes = []
        
        for pred, gt in zip(predictions, ground_truths):
            # 预测框
            if len(pred['boxes']) > 0:
                pred_mask = pred['classes'] == class_id
                if isinstance(pred_mask, np.ndarray) and pred_mask.any():
                    pred_scores.extend(pred['scores'][pred_mask])
                    pred_boxes.extend(pred['boxes'][pred_mask])
            
            # 真实标注
            if len(gt['boxes']) > 0:
                gt_mask = gt['classes'] == class_id
                if isinstance(gt_mask, np.ndarray) and gt_mask.any():
                    gt_boxes.extend(gt['boxes'][gt_mask])
                elif isinstance(gt_mask, bool) and gt_mask:
                    gt_boxes.extend(gt['boxes'])
        
        if len(pred_boxes) == 0 or len(gt_boxes) == 0:
            aps.append(0.0)
            continue
        
        # 计算AP (简化版本)
        ap = _compute_ap(pred_boxes, pred_scores, gt_boxes, iou_threshold)
        aps.append(ap)
    
    return np.mean(aps)


def _compute_ap(
    pred_boxes: List[np.ndarray],
    pred_scores: List[float],
    gt_boxes: List[np.ndarray],
    iou_threshold: float
) -> float:
    """计算单个类别的AP (简化版本)"""
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return 0.0
    
    # 按置信度排序
    sorted_indices = np.argsort(pred_scores)[::-1]
    pred_boxes = [pred_boxes[i] for i in sorted_indices]
    
    # 匹配预测框和真实框
    matched_gt = set()
    true_positives = []
    false_positives = []
    
    for pred_box in pred_boxes:
        best_iou = 0
        best_gt_idx = -1
        
        for gt_idx, gt_box in enumerate(gt_boxes):
            if gt_idx in matched_gt:
                continue
            
            iou = compute_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        if best_iou >= iou_threshold:
            true_positives.append(1)
            false_positives.append(0)
            matched_gt.add(best_gt_idx)
        else:
            true_positives.append(0)
            false_positives.append(1)
    
    # 计算精确率-召回率曲线
    tp_cumsum = np.cumsum(true_positives)
    fp_cumsum = np.cumsum(false_positives)
    
    recalls = tp_cumsum / len(gt_boxes)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
    
    # 计算AP (使用11点插值)
    ap = 0
    for t in np.arange(0, 1.1, 0.1):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 11
    
    return ap


def print_dota_metrics(metrics: Dict):
    """
    打印DOTA评估指标
    
    Args:
        metrics: 评估指标字典
    """
    print("\n" + "="*80)
    print("DOTA EVALUATION METRICS")
    print("="*80)
    
    print(f"mAP: {metrics['map']:.4f}")
    print(f"Mean Precision: {metrics['mean_precision']:.4f}")
    print(f"Mean Recall: {metrics['mean_recall']:.4f}")
    print(f"Mean F1: {metrics['mean_f1']:.4f}")
    
    print("\nPer-class metrics:")
    print("-" * 80)
    print(f"{'Class':<20} {'Precision':<10} {'Recall':<10} {'F1':<10}")
    print("-" * 80)
    
    for class_name, class_metrics in metrics['class_metrics'].items():
        print(f"{class_name:<20} {class_metrics['precision']:<10.4f} "
              f"{class_metrics['recall']:<10.4f} {class_metrics['f1']:<10.4f}")
    
    print("="*80)


class DOTAMetrics:
    """DOTA指标计算类"""
    
    def __init__(self):
        self.standard_metrics = ['mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l']
        self.agent_specific_metrics = [
            'query_refinement_gain',
            'vlm_verification_accuracy', 
            'iteration_efficiency',
            'false_positive_reduction_rate',
            'occlusion_handling_improvement'
        ]
    
    def evaluate_agentic_improvement(self, baseline_results, agentic_results):
        """评估Agentic方法的改进"""
        improvements = {}
        
        for metric in self.standard_metrics:
            improvements[f'{metric}_improvement'] = (
                agentic_results[metric] - baseline_results[metric]
            )
        
        # 计算文章中提到的主要指标
        improvements['false_positive_reduction'] = (
            (baseline_results['false_positives'] - agentic_results['false_positives']) 
            / baseline_results['false_positives']
        )
        
        return improvements