"""统一模型加载器"""
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from src.models.yolo_wrapper import YOLOWrapper
from src.models.vlm_wrapper import VLMWrapper

logger = logging.getLogger(__name__)


class ModelLoader:
    """统一的模型加载器类"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化模型加载器
        
        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path) if config_path else {}
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        logger.info(f"Loading config from {config_path}")
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    
    def load_yolo(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs
    ) -> YOLOWrapper:
        """
        加载YOLO模型
        
        Args:
            model_path: 模型路径(可选,从配置读取)
            device: 设备(可选,从配置读取)
            **kwargs: 其他参数
            
        Returns:
            YOLOWrapper实例
        """
        # 从配置或参数获取设置
        yolo_config = self.config.get('yolo', {})
        
        model_path = model_path or yolo_config.get('model_name', 'yolov8s.pt')
        device = device or yolo_config.get('device', 'cuda')
        conf_threshold = kwargs.get('conf_threshold', yolo_config.get('confidence_threshold', 0.25))
        iou_threshold = kwargs.get('iou_threshold', yolo_config.get('iou_threshold', 0.45))
        imgsz = kwargs.get('imgsz', yolo_config.get('imgsz', 1024))
        
        logger.info(f"Loading YOLO model: {model_path}")
        yolo_model = YOLOWrapper(
            model_path=model_path,
            device=device,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            imgsz=imgsz
        )
        
        return yolo_model
    
    def load_vlm(
        self,
        model_type: Optional[str] = None,
        model_name: Optional[str] = None,
        device: Optional[str] = None
    ) -> VLMWrapper:
        """
        加载VLM模型
        
        Args:
            model_type: 模型类型(可选,从配置读取)
            model_name: 模型名称(可选,从配置读取)
            device: 设备(可选,从配置读取)
            
        Returns:
            VLMWrapper实例
        """
        # 从配置或参数获取设置
        vlm_config = self.config.get('vlm', {})
        
        model_type = model_type or vlm_config.get('model_type', 'clip')
        model_name = model_name or vlm_config.get('model_name', 'ViT-B/32')
        device = device or vlm_config.get('device', 'cuda')
        
        logger.info(f"Loading VLM model: {model_type}/{model_name}")
        vlm_model = VLMWrapper(
            model_type=model_type,
            model_name=model_name,
            device=device
        )
        
        return vlm_model
    
    def load_all_models(self) -> Dict[str, Any]:
        """
        加载所有模型
        
        Returns:
            包含所有模型的字典
        """
        models = {
            'yolo': self.load_yolo(),
            'vlm': self.load_vlm()
        }
        
        logger.info("All models loaded successfully")
        return models