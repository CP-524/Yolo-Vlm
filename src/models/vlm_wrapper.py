"""VLM模型封装（CLIP、LLaVA等）"""
import torch
import clip
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)


class VLMWrapper:
    """视觉语言模型封装类"""
    
    def __init__(
        self,
        model_type: str = "clip",
        model_name: str = "ViT-B/32",
        device: str = "cuda"
    ):
        """
        初始化VLM模型
        
        Args:
            model_type: 模型类型 (clip, llava, blip2)
            model_name: 模型名称
            device: 设备
        """
        self.model_type = model_type.lower()
        self.model_name = model_name
        self.device = device
        
        if self.model_type == "clip":
            self._load_clip()
        elif self.model_type == "llava":
            self._load_llava()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def _load_clip(self):
        """加载CLIP模型"""
        logger.info(f"Loading CLIP model: {self.model_name}")
        self.model, self.preprocess = clip.load(self.model_name, device=self.device)
        self.model.eval()
        logger.info("CLIP model loaded successfully")
    
    def _load_llava(self):
        """加载LLaVA模型（待实现）"""
        raise NotImplementedError("LLaVA support is not yet implemented")
    
    @torch.no_grad()
    def encode_text(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """
        编码文本
        
        Args:
            texts: 文本或文本列表
            
        Returns:
            文本特征张量
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Tokenize文本
        text_tokens = clip.tokenize(texts).to(self.device)
        
        # 编码
        text_features = self.model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features
    
    @torch.no_grad()
    def encode_image(self, images: Union[str, Image.Image, List]) -> torch.Tensor:
        """
        编码图像
        
        Args:
            images: 图像路径、PIL图像或列表
            
        Returns:
            图像特征张量
        """
        # 处理输入
        if isinstance(images, str):
            images = [Image.open(images)]
        elif isinstance(images, Image.Image):
            images = [images]
        elif isinstance(images, list) and isinstance(images[0], str):
            images = [Image.open(img) for img in images]
        
        # 预处理图像
        image_inputs = torch.stack([self.preprocess(img) for img in images]).to(self.device)
        
        # 编码
        image_features = self.model.encode_image(image_inputs)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features
    
    @torch.no_grad()
    def compute_similarity(
        self, 
        images: Union[str, Image.Image, List],
        texts: Union[str, List[str]]
    ) -> np.ndarray:
        """
        计算图像-文本相似度
        
        Args:
            images: 图像
            texts: 文本
            
        Returns:
            相似度矩阵 [num_images, num_texts]
        """
        image_features = self.encode_image(images)
        text_features = self.encode_text(texts)
        
        # 计算余弦相似度
        similarity = (image_features @ text_features.T).cpu().numpy()
        
        return similarity
    
    @torch.no_grad()
    def verify_detection(
        self,
        image: Union[str, Image.Image],
        query: str,
        threshold: float = 0.3
    ) -> Tuple[bool, float]:
        """
        验证检测结果
        
        Args:
            image: 图像
            query: 查询文本
            threshold: 相似度阈值
            
        Returns:
            (是否匹配, 相似度分数)
        """
        similarity = self.compute_similarity(image, query)[0, 0]
        is_match = similarity >= threshold
        
        return is_match, float(similarity)
    
    @torch.no_grad()
    def batch_verify(
        self,
        images: List[Union[str, Image.Image]],
        queries: List[str],
        threshold: float = 0.3
    ) -> List[Tuple[bool, float]]:
        """
        批量验证检测结果
        
        Args:
            images: 图像列表
            queries: 查询文本列表
            threshold: 相似度阈值
            
        Returns:
            验证结果列表
        """
        if len(images) != len(queries):
            raise ValueError("Number of images and queries must match")
        
        results = []
        for img, query in zip(images, queries):
            is_match, score = self.verify_detection(img, query, threshold)
            results.append((is_match, score))
        
        return results
    
    def get_top_k_classes(
        self,
        image: Union[str, Image.Image],
        class_names: List[str],
        k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        获取最相似的top-k类别
        
        Args:
            image: 图像
            class_names: 类别名称列表
            k: 返回top-k结果
            
        Returns:
            [(类别名, 相似度分数), ...]
        """
        # 构建文本提示
        texts = [f"a photo of a {name}" for name in class_names]
        
        # 计算相似度
        similarities = self.compute_similarity(image, texts)[0]
        
        # 获取top-k
        top_k_indices = np.argsort(similarities)[::-1][:k]
        results = [(class_names[i], float(similarities[i])) for i in top_k_indices]
        
        return results