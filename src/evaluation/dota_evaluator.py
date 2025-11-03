"""DOTA数据集评估器"""
import logging
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
import numpy as np

from src.agents.detection_agent import DetectionAgent
from src.utils.data_utils import load_annotations, get_image_list, yolo_to_xyxy, obb_to_poly
from src.evaluation.dota_metrics import compute_dota_metrics, print_dota_metrics
from src.evaluation.obb_metrics import compute_obb_precision_recall, compute_obb_map
from PIL import Image

logger = logging.getLogger(__name__)


class DOTAEvaluator:
    """DOTA数据集评估器"""
    
    def __init__(
        self,
        detection_agent: DetectionAgent,
        data_root: str,
        class_names: List[str],
        split: str = 'val'
    ):
        """
        初始化评估器
        
        Args:
            detection_agent: 检测Agent
            data_root: 数据集根目录
            class_names: 类别名称列表
            split: 数据集划分 (train/val/test)
        """
        self.agent = detection_agent
        self.data_root = Path(data_root)
        self.class_names = class_names
        self.split = split
        
        # 数据路径
        self.image_dir = self.data_root / 'images' / split
        self.label_dir = self.data_root / 'labels' / split
        
        logger.info(f"Initialized DOTA Evaluator - Split: {split}")
        logger.info(f"Image dir: {self.image_dir}")
        logger.info(f"Label dir: {self.label_dir}")
    
    def _generate_query_from_gt(self, gt_classes: np.ndarray) -> Optional[str]:
        """
        根据真实标注的类别生成查询
        
        Args:
            gt_classes: 真实类别ID数组
            
        Returns:
            生成的查询字符串
        """
        if len(gt_classes) == 0:
            return None
        
        # 获取唯一的类别
        unique_classes = np.unique(gt_classes)
        class_names_in_image = [self.class_names[int(cls)] for cls in unique_classes]
        
        # 生成查询
        if len(class_names_in_image) == 1:
            query = f"find all {class_names_in_image[0]}"
        elif len(class_names_in_image) == 2:
            query = f"find all {class_names_in_image[0]} and {class_names_in_image[1]}"
        else:
            # 多个类别，使用逗号分隔
            query = f"find all {', '.join(class_names_in_image[:-1])} and {class_names_in_image[-1]}"
        
        return query
    
    def evaluate(
        self,
        use_agentic: bool = True,
        query: Optional[str] = None,
        max_images: Optional[int] = None,
        iou_threshold: float = 0.5,
        use_auto_query: bool = False
    ) -> Dict:
        """
        评估检测性能
        
        Args:
            use_agentic: 是否使用Agentic方法
            query: 查询文本(可选，如果提供则所有图像使用相同查询)
            max_images: 最大评估图像数(可选)
            iou_threshold: IoU阈值
            use_auto_query: 是否根据GT自动生成查询(仅在use_agentic=True时有效)
            
        Returns:
            评估指标字典
        """
        # 获取图像列表
        image_files = get_image_list(self.image_dir)
        
        if max_images:
            image_files = image_files[:max_images]
        
        logger.info(f"Evaluating on {len(image_files)} images")
        if use_agentic and use_auto_query:
            logger.info("Auto-query mode enabled: generating queries from ground truth")
        
        predictions = []
        ground_truths = []
        
        # 遍历图像进行检测
        for img_path in tqdm(image_files, desc="Evaluating"):
            # 加载标注（提前加载以支持自动查询）
            label_path = self.label_dir / f"{img_path.stem}.txt"
            gt_ann = load_annotations(str(label_path), format='yolo')
            
            # 获取图像尺寸(从标注推断或快速读取)
            # 使用PIL快速获取图像尺寸,不解码整个图像
            with Image.open(img_path) as image:
                img_width, img_height = image.size
            
            # 确定使用的查询
            current_query = query
            if use_agentic and use_auto_query and query is None:
                current_query = self._generate_query_from_gt(gt_ann['classes'])
            
            # 执行检测
            if use_agentic:
                # Agentic方法: YOLO + 查询过滤 + VLM验证
                pred = self.agent.detect(str(img_path), query=current_query)
            else:
                # Baseline方法: 仅YOLO检测
                pred = self.agent.baseline_detect(str(img_path))
            
            predictions.append(pred)
            
            # 处理真实标注
            is_obb = gt_ann.get('is_obb', False)
            
            if is_obb and len(gt_ann.get('obb_boxes', [])) > 0:
                # OBB格式 - 将归一化的多边形坐标转换为绝对坐标
                obb_boxes_abs = []
                for obb in gt_ann['obb_boxes']:
                    obb_abs = obb_to_poly(obb, img_width, img_height)
                    obb_boxes_abs.append(obb_abs)
                
                # 同时转换水平框用于兼容
                if len(gt_ann['boxes']) > 0:
                    gt_boxes = yolo_to_xyxy(gt_ann['boxes'], img_width, img_height)
                else:
                    gt_boxes = np.array([])
                
                gt = {
                    'boxes': gt_boxes,
                    'obb_boxes': np.array(obb_boxes_abs),
                    'classes': gt_ann['classes'],
                    'is_obb': True
                }
            else:
                # 标准水平框格式
                if len(gt_ann['boxes']) > 0:
                    gt_boxes = yolo_to_xyxy(gt_ann['boxes'], img_width, img_height)
                else:
                    gt_boxes = np.array([])
                
                gt = {
                    'boxes': gt_boxes,
                    'classes': gt_ann['classes'],
                    'is_obb': False
                }
            
            ground_truths.append(gt)
        
        # 检查是否使用OBB
        use_obb = any(gt.get('is_obb', False) for gt in ground_truths)
        
        # 计算指标
        if use_obb:
            logger.info("Using OBB evaluation metrics")
            # 计算每个类别的精确率和召回率
            class_metrics = compute_obb_precision_recall(
                predictions,
                ground_truths,
                self.class_names,
                iou_threshold
            )
            
            # 计算mAP
            map_results = compute_obb_map(
                predictions,
                ground_truths,
                self.class_names
            )
            
            # 计算平均指标
            precisions = [m['precision'] for m in class_metrics.values()]
            recalls = [m['recall'] for m in class_metrics.values()]
            f1_scores = [m['f1'] for m in class_metrics.values()]
            
            metrics = {
                'map': map_results['mAP'],
                'map50': map_results['mAP50'],
                'map75': map_results['mAP75'],
                'mean_precision': np.mean(precisions),
                'mean_recall': np.mean(recalls),
                'mean_f1': np.mean(f1_scores),
                'precision': precisions,
                'recall': recalls,
                'f1': f1_scores,
                'class_metrics': class_metrics,
                'use_obb': True
            }
        else:
            logger.info("Using standard horizontal box evaluation metrics")
            metrics = compute_dota_metrics(
                predictions,
                ground_truths,
                self.class_names,
                iou_threshold
            )
            metrics['use_obb'] = False
        
        # 打印结果
        self._print_metrics(metrics)
        
        return metrics
    
    def _print_metrics(self, metrics: Dict):
        """打印评估指标"""
        print("\n" + "="*80)
        print("DOTA EVALUATION METRICS")
        if metrics.get('use_obb', False):
            print("(Using OBB / Oriented Bounding Box)")
        print("="*80)
        
        # 打印主要指标
        if 'map50' in metrics:
            print(f"mAP@50:     {metrics['map50']:.4f}")
        if 'map75' in metrics:
            print(f"mAP@75:     {metrics['map75']:.4f}")
        print(f"mAP:        {metrics['map']:.4f}")
        print(f"Precision:  {metrics['mean_precision']:.4f}")
        print(f"Recall:     {metrics['mean_recall']:.4f}")
        print(f"F1:         {metrics['mean_f1']:.4f}")
        
        # 打印每个类别的指标
        print("\nPer-class metrics:")
        print("-" * 80)
        print(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Pred/GT'}")
        print("-" * 80)
        
        for class_name, class_metrics in metrics['class_metrics'].items():
            num_pred = class_metrics.get('num_pred', 0)
            num_gt = class_metrics.get('num_gt', 0)
            print(f"{class_name:<20} {class_metrics['precision']:<12.4f} "
                  f"{class_metrics['recall']:<12.4f} {class_metrics['f1']:<12.4f} "
                  f"{num_pred}/{num_gt}")
        
        print("="*80)
    
    def compare_methods(
        self,
        query: Optional[str] = None,
        max_images: Optional[int] = None,
        use_auto_query: bool = False
    ) -> Dict:
        """
        对比基线和Agentic方法
        
        Args:
            query: 查询文本
            max_images: 最大评估图像数
            use_auto_query: 是否为Agentic方法使用自动查询
            
        Returns:
            对比结果字典
        """
        logger.info("Comparing baseline and agentic methods")
        if use_auto_query:
            logger.info("Auto-query enabled for Agentic method")
        
        # 评估基线方法
        print("\n" + "="*80)
        print("BASELINE METHOD (YOLO Only)")
        print("="*80)
        baseline_metrics = self.evaluate(
            use_agentic=False,
            max_images=max_images
        )
        
        # 评估Agentic方法
        print("\n" + "="*80)
        if use_auto_query:
            print("AGENTIC METHOD (with Auto-Generated Queries)")
        else:
            print("AGENTIC METHOD")
        print("="*80)
        agentic_metrics = self.evaluate(
            use_agentic=True,
            query=query,
            max_images=max_images,
            use_auto_query=use_auto_query
        )
        
        # 计算改进
        improvement = {
            'mAP': agentic_metrics['map'] - baseline_metrics['map'],
            'precision': agentic_metrics['mean_precision'] - baseline_metrics['mean_precision'],
            'recall': agentic_metrics['mean_recall'] - baseline_metrics['mean_recall'],
            'f1': agentic_metrics['mean_f1'] - baseline_metrics['mean_f1']
        }
        
        print("\n" + "="*80)
        print("IMPROVEMENT (Agentic - Baseline)")
        print("="*80)
        for metric, value in improvement.items():
            sign = "+" if value >= 0 else ""
            print(f"{metric:<12} {sign}{value:.4f}")
        print("="*80 + "\n")
        
        return {
            'baseline': baseline_metrics,
            'agentic': agentic_metrics,
            'improvement': improvement
        }