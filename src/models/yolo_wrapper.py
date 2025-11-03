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
        
        # logger.info(f"Loading YOLO model from {model_path}")
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
            verbose=False,
            stream=False,  # 单图像不使用stream模式
            half=True  # 使用FP16加速(如果GPU支持)
        )
        
        return self._parse_results(results[0])
    
    def predict_batch(
        self, 
        image_paths: List[str],
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        batch_size: int = 1
    ) -> List[Dict]:
        """
        批量检测
        
        Args:
            image_paths: 图像路径列表
            conf: 置信度阈值
            iou: IOU阈值
            batch_size: 批处理大小(增大可提速,但占用更多显存)
            
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
            stream=True,
            batch=batch_size,  # 使用批处理
            half=True  # 使用FP16加速
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
        # 检查是否有OBB结果
        if hasattr(result, 'obb') and result.obb is not None:
            return self._parse_obb_results(result)
        
        # 处理普通水平框
        boxes = result.boxes
        
        if boxes is None or len(boxes) == 0:
            return {
                "boxes": np.array([]),
                "scores": np.array([]),
                "classes": np.array([]),
                "class_names": [],
                "is_obb": False
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
            "image_shape": result.orig_shape,
            "is_obb": False
        }
    
    def _parse_obb_results(self, result) -> Dict:
        """
        解析YOLO OBB检测结果
        
        Args:
            result: YOLO检测结果对象
            
        Returns:
            格式化的检测结果字典
        """
        obb = result.obb
        
        if obb is None or len(obb) == 0:
            return {
                "boxes": np.array([]),
                "obb_boxes": np.array([]),
                "scores": np.array([]),
                "classes": np.array([]),
                "class_names": [],
                "is_obb": True
            }
        
        # 提取OBB框坐标 (x, y, w, h, angle)
        if hasattr(obb, 'xywhr'):
            obb_coords = obb.xywhr.cpu().numpy()  # [cx, cy, w, h, rotation]
        else:
            obb_coords = obb.data.cpu().numpy()[:, :5]  # 前5列是坐标
        
        # 提取置信度和类别
        scores = obb.conf.cpu().numpy()
        classes = obb.cls.cpu().numpy().astype(int)
        
        # 获取类别名称
        class_names = [result.names[cls] for cls in classes]
        
        # 将OBB转换为水平框(用于兼容)
        xyxy = self._obb_to_xyxy(obb_coords)
        
        # 将OBB (cx, cy, w, h, angle) 转换为8点多边形格式
        obb_poly = self._xywhr_to_poly(obb_coords)
        
        return {
            "boxes": xyxy,  # 水平框(兼容)
            "obb_boxes": obb_poly,  # OBB框 [x1, y1, x2, y2, x3, y3, x4, y4]
            "scores": scores,
            "classes": classes,
            "class_names": class_names,
            "image_shape": result.orig_shape,
            "is_obb": True
        }
    
    def _xywhr_to_poly(self, xywhr_array: np.ndarray) -> np.ndarray:
        """
        将xywhr格式的OBB批量转换为8点多边形格式
        
        Args:
            xywhr_array: [N, 5] (cx, cy, w, h, angle)
            
        Returns:
            poly_array: [N, 8] (x1, y1, x2, y2, x3, y3, x4, y4)
        """
        if len(xywhr_array) == 0:
            return np.array([])
        
        poly_list = []
        for xywhr in xywhr_array:
            cx, cy, w, h, angle = xywhr
            
            # 计算四个角点(未旋转时)
            corners = np.array([
                [-w/2, -h/2],  # 左上
                [w/2, -h/2],   # 右上  
                [w/2, h/2],    # 右下
                [-w/2, h/2]    # 左下
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
            
            # 展平为8点格式
            poly_list.append(rotated_corners.flatten())
        
        return np.array(poly_list)
    
    def _obb_to_xyxy(self, obb_coords: np.ndarray) -> np.ndarray:
        """
        将OBB格式转换为水平边界框(xyxy格式)
        
        Args:
            obb_coords: OBB坐标 [cx, cy, w, h, angle]
            
        Returns:
            水平边界框 [x1, y1, x2, y2]
        """
        if len(obb_coords) == 0:
            return np.array([])
        
        xyxy = []
        for obb in obb_coords:
            cx, cy, w, h = obb[:4]
            # 简单近似：使用旋转矩形的外接矩形
            # 对于更精确的转换，需要考虑旋转角度
            if len(obb) > 4:
                angle = obb[4]
                # 计算旋转后的边界
                cos_a = np.abs(np.cos(angle))
                sin_a = np.abs(np.sin(angle))
                w_bbox = w * cos_a + h * sin_a
                h_bbox = w * sin_a + h * cos_a
            else:
                w_bbox, h_bbox = w, h
            
            x1 = cx - w_bbox / 2
            y1 = cy - h_bbox / 2
            x2 = cx + w_bbox / 2
            y2 = cy + h_bbox / 2
            
            xyxy.append([x1, y1, x2, y2])
        
        return np.array(xyxy)
    
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