"""YOLO模型封装"""
import torch
from ultralytics import YOLO
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class YOLOWrapper:
    """YOLO模型封装类，提供统一的检测接口"""
    
    def __init__(
        self, 
        model_path: str = "yolov8s.pt",
        device: str = "cuda",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        imgsz: int = 1024
    ):
        """
        初始化YOLO模型
        
        Args:
            model_path: 模型权重路径
            device: 设备 (cuda/cpu)
            conf_threshold: 置信度阈值
            iou_threshold: IOU阈值
            imgsz: 输入图像大小
        """
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        
        logger.info(f"Loading YOLO model from {model_path}")
        self.model = YOLO(model_path)
        self.model.to(device)
        logger.info(f"YOLO model loaded successfully on {device}")
        
    def predict(
        self, 
        image_path: str,
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        classes: Optional[List[int]] = None
    ) -> Dict:
        """
        对单张图像进行检测
        
        Args:
            image_path: 图像路径
            conf: 置信度阈值(可选)
            iou: IOU阈值(可选)
            classes: 要检测的类别列表(可选)
            
        Returns:
            检测结果字典
        """
        conf = conf or self.conf_threshold
        iou = iou or self.iou_threshold
        
        results = self.model.predict(
            source=image_path,
            conf=conf,
            iou=iou,
            imgsz=self.imgsz,
            device=self.device,
            classes=classes,
            verbose=False
        )
        
        return self._parse_results(results[0])
    
    def predict_batch(
        self, 
        image_paths: List[str],
        conf: Optional[float] = None,
        iou: Optional[float] = None
    ) -> List[Dict]:
        """
        批量检测
        
        Args:
            image_paths: 图像路径列表
            conf: 置信度阈值
            iou: IOU阈值
            
        Returns:
            检测结果列表
        """
        conf = conf or self.conf_threshold
        iou = iou or self.iou_threshold
        
        results = self.model.predict(
            source=image_paths,
            conf=conf,
            iou=iou,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
            stream=True
        )
        
        return [self._parse_results(r) for r in results]
    
    def _parse_results(self, result) -> Dict:
        """
        解析YOLO检测结果
        
        Args:
            result: YOLO检测结果对象
            
        Returns:
            格式化的检测结果字典
        """
        boxes = result.boxes
        
        if boxes is None or len(boxes) == 0:
            return {
                "boxes": np.array([]),
                "scores": np.array([]),
                "classes": np.array([]),
                "class_names": []
            }
        
        # 提取边界框、置信度和类别
        xyxy = boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
        scores = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)
        
        # 获取类别名称
        class_names = [result.names[cls] for cls in classes]
        
        return {
            "boxes": xyxy,
            "scores": scores,
            "classes": classes,
            "class_names": class_names,
            "image_shape": result.orig_shape
        }
    
    def train(self, data_yaml: str, epochs: int = 100, **kwargs):
        """
        训练YOLO模型
        
        Args:
            data_yaml: 数据集配置文件路径
            epochs: 训练轮数
            **kwargs: 其他训练参数
        """
        logger.info(f"Starting training for {epochs} epochs")
        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=self.imgsz,
            device=self.device,
            **kwargs
        )
        return results
    
    def val(self, data_yaml: str, **kwargs):
        """
        验证YOLO模型
        
        Args:
            data_yaml: 数据集配置文件路径
            **kwargs: 其他验证参数
        """
        logger.info("Starting validation")
        results = self.model.val(
            data=data_yaml,
            imgsz=self.imgsz,
            device=self.device,
            **kwargs
        )
        return results
    
    def export(self, format: str = "onnx", **kwargs):
        """
        导出模型
        
        Args:
            format: 导出格式 (onnx, torchscript等)
            **kwargs: 其他导出参数
        """
        logger.info(f"Exporting model to {format} format")
        return self.model.export(format=format, **kwargs)