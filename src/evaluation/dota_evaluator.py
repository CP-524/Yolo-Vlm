"""DOTA数据集评估器"""
import logging
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
import numpy as np

from src.agents.detection_agent import DetectionAgent
from src.utils.data_utils import load_annotations, get_image_list, yolo_to_xyxy
from src.evaluation.dota_metrics import compute_dota_metrics, print_dota_metrics
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
    
    def evaluate(
        self,
        use_agentic: bool = True,
        query: Optional[str] = None,
        max_images: Optional[int] = None,
        iou_threshold: float = 0.5
    ) -> Dict:
        """
        评估检测性能
        
        Args:
            use_agentic: 是否使用Agentic方法
            query: 查询文本(可选)
            max_images: 最大评估图像数(可选)
            iou_threshold: IoU阈值
            
        Returns:
            评估指标字典
        """
        # 获取图像列表
        image_files = get_image_list(self.image_dir)
        
        if max_images:
            image_files = image_files[:max_images]
        
        logger.info(f"Evaluating on {len(image_files)} images")
        
        predictions = []
        ground_truths = []
        
        # 遍历图像进行检测
        for img_path in tqdm(image_files, desc="Evaluating"):
            # 加载图像获取尺寸
            image = Image.open(img_path)
            img_width, img_height = image.size
            
            # 执行检测
            if use_agentic:
                pred = self.agent.detect(str(img_path), query=query)
            else:
                pred = self.agent.baseline_detect(str(img_path))
            
            predictions.append(pred)
            
            # 加载标注
            label_path = self.label_dir / f"{img_path.stem}.txt"
            gt_ann = load_annotations(str(label_path), format='yolo')
            
            # 转换YOLO格式到xyxy
            if len(gt_ann['boxes']) > 0:
                gt_boxes = yolo_to_xyxy(gt_ann['boxes'], img_width, img_height)
            else:
                gt_boxes = np.array([])
            
            gt = {
                'boxes': gt_boxes,
                'classes': gt_ann['classes']
            }
            ground_truths.append(gt)
        
        # 计算指标
        metrics = compute_dota_metrics(
            predictions,
            ground_truths,
            self.class_names,
            iou_threshold
        )
        
        # 打印结果
        print_dota_metrics(metrics)
        
        return metrics
    
    def compare_methods(
        self,
        query: Optional[str] = None,
        max_images: Optional[int] = None
    ) -> Dict:
        """
        对比基线和Agentic方法
        
        Args:
            query: 查询文本
            max_images: 最大评估图像数
            
        Returns:
            对比结果字典
        """
        logger.info("Comparing baseline and agentic methods")
        
        # 评估基线方法
        print("\n" + "="*80)
        print("BASELINE METHOD")
        print("="*80)
        baseline_metrics = self.evaluate(
            use_agentic=False,
            max_images=max_images
        )
        
        # 评估Agentic方法
        print("\n" + "="*80)
        print("AGENTIC METHOD")
        print("="*80)
        agentic_metrics = self.evaluate(
            use_agentic=True,
            query=query,
            max_images=max_images
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