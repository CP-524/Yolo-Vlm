"""可视化工具"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, List, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class Visualizer:
    """检测结果可视化工具"""
    
    def __init__(self, class_names: Optional[List[str]] = None):
        """
        初始化可视化器
        
        Args:
            class_names: 类别名称列表
        """
        self.class_names = class_names
        self.colors = self._generate_colors(len(class_names) if class_names else 15)
    
    def _generate_colors(self, num_classes: int) -> List[Tuple[int, int, int]]:
        """生成随机颜色"""
        np.random.seed(42)
        colors = []
        for _ in range(num_classes):
            color = tuple(np.random.randint(0, 255, 3).tolist())
            colors.append(color)
        return colors
    
    def draw_detections(
        self,
        image_path: str,
        detections: Dict,
        output_path: Optional[str] = None,
        show: bool = False,
        thickness: int = 2,
        font_scale: float = 0.5
    ) -> np.ndarray:
        """
        在图像上绘制检测框
        
        Args:
            image_path: 输入图像路径
            detections: 检测结果
            output_path: 输出路径(可选)
            show: 是否显示
            thickness: 线条粗细
            font_scale: 字体大小
            
        Returns:
            绘制后的图像
        """
        # 读取图像
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # 绘制每个检测框
        for i, box in enumerate(detections['boxes']):
            x1, y1, x2, y2 = map(int, box)
            class_id = int(detections['classes'][i])
            score = float(detections['scores'][i])
            
            # 选择颜色
            color = self.colors[class_id]
            
            # 绘制边界框
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
            
            # 准备标签
            class_name = detections['class_names'][i] if 'class_names' in detections else str(class_id)
            label = f"{class_name}: {score:.2f}"
            
            # 添加VLM分数(如果有)
            if 'vlm_scores' in detections and i < len(detections['vlm_scores']):
                vlm_score = float(detections['vlm_scores'][i])
                label += f" (VLM: {vlm_score:.2f})"
            
            # 绘制标签背景
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            cv2.rectangle(image, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
            
            # 绘制标签文本
            cv2.putText(
                image,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                thickness
            )
        
        # 保存图像
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(output_path, image)
            logger.info(f"Saved visualization to {output_path}")
        
        # 显示图像
        if show:
            plt.figure(figsize=(12, 8))
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.title(f"Detections: {len(detections['boxes'])} objects")
            plt.show()
        
        return image
    
    def compare_results(
        self,
        image_path: str,
        baseline_results: Dict,
        agentic_results: Dict,
        output_path: Optional[str] = None,
        show: bool = False
    ):
        """
        对比基线和Agentic方法的结果
        
        Args:
            image_path: 图像路径
            baseline_results: 基线结果
            agentic_results: Agentic结果
            output_path: 输出路径
            show: 是否显示
        """
        # 读取图像
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # 创建两个副本
        img_baseline = image.copy()
        img_agentic = image.copy()
        
        # 绘制基线结果
        for i, box in enumerate(baseline_results['boxes']):
            x1, y1, x2, y2 = map(int, box)
            class_id = int(baseline_results['classes'][i])
            color = self.colors[class_id]
            cv2.rectangle(img_baseline, (x1, y1), (x2, y2), color, 2)
        
        # 绘制Agentic结果
        for i, box in enumerate(agentic_results['boxes']):
            x1, y1, x2, y2 = map(int, box)
            class_id = int(agentic_results['classes'][i])
            color = self.colors[class_id]
            cv2.rectangle(img_agentic, (x1, y1), (x2, y2), color, 2)
        
        # 并排显示
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        
        axes[0].imshow(cv2.cvtColor(img_baseline, cv2.COLOR_BGR2RGB))
        axes[0].set_title(f"Baseline ({len(baseline_results['boxes'])} objects)")
        axes[0].axis('off')
        
        axes[1].imshow(cv2.cvtColor(img_agentic, cv2.COLOR_BGR2RGB))
        axes[1].set_title(f"Agentic ({len(agentic_results['boxes'])} objects)")
        axes[1].axis('off')
        
        plt.tight_layout()
        
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved comparison to {output_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_metrics(
        self,
        metrics: Dict,
        output_path: Optional[str] = None,
        show: bool = False
    ):
        """
        绘制评估指标
        
        Args:
            metrics: 指标字典
            output_path: 输出路径
            show: 是否显示
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Precision
        if 'precision' in metrics:
            axes[0, 0].bar(range(len(metrics['precision'])), metrics['precision'])
            axes[0, 0].set_title('Precision per Class')
            axes[0, 0].set_xlabel('Class ID')
            axes[0, 0].set_ylabel('Precision')
        
        # Recall
        if 'recall' in metrics:
            axes[0, 1].bar(range(len(metrics['recall'])), metrics['recall'])
            axes[0, 1].set_title('Recall per Class')
            axes[0, 1].set_xlabel('Class ID')
            axes[0, 1].set_ylabel('Recall')
        
        # F1 Score
        if 'f1' in metrics:
            axes[1, 0].bar(range(len(metrics['f1'])), metrics['f1'])
            axes[1, 0].set_title('F1 Score per Class')
            axes[1, 0].set_xlabel('Class ID')
            axes[1, 0].set_ylabel('F1 Score')
        
        # mAP
        if 'map' in metrics:
            axes[1, 1].text(0.5, 0.5, f"mAP: {metrics['map']:.4f}", 
                           ha='center', va='center', fontsize=20)
            axes[1, 1].set_title('Overall Performance')
            axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved metrics plot to {output_path}")
        
        if show:
            plt.show()
        else:
            plt.close()