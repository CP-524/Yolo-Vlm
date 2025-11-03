import torch
from ultralytics import YOLO
import clip
import json
from PIL import Image
import sys
import os

# 添加项目路径以便导入
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.model_loader import ModelLoader
from src.agents.detection_agent import DetectionAgent
from src.agents.vlm_agent import VLMAgent
from src.agents.query_processor import QueryProcessor
from src.utils.visualization import Visualizer
import yaml

class AgenticDetectionPipeline:
    """
    Agentic目标检测Pipeline (使用项目已有CLIP架构)
    核心思想：检测-批判-验证-迭代
    """
    
    def __init__(self, 
                 detector_model="YOLO/DOTA.pt", 
                 config_path="configs/model_configs.yaml"):
        """
        初始化Pipeline
        
        参数：
            detector_model: YOLO模型路径
            config_path: 模型配置文件路径
        """
        # 加载类别名称
        with open('configs/DOTA.yaml', 'r', encoding='utf-8') as f:
            data_config = yaml.safe_load(f)
        self.class_names = list(data_config['names'].values())
        
        # 使用项目的ModelLoader
        model_loader = ModelLoader(config_path)
        self.detector = model_loader.load_yolo(device='cuda')
        print(f"✓ 检测器加载完成: {detector_model}")
        
        # 加载CLIP VLM (快速，只需1-2秒)
        vlm_model = model_loader.load_vlm(device='cuda')
        self.vlm_agent = VLMAgent(vlm_model, verification_threshold=0.15)
        print(f"✓ CLIP VLM加载完成")
        
        # 创建查询处理器
        self.query_processor = QueryProcessor(self.class_names)
        print(f"✓ 查询处理器初始化完成")
        
        # 创建完整DetectionAgent
        self.agent = DetectionAgent(
            yolo_model=self.detector,
            vlm_agent=self.vlm_agent,
            query_processor=self.query_processor,
            enable_vlm=True,
            enable_query_processing=True
        )
        
        # 可视化工具
        self.visualizer = Visualizer(self.class_names)
        
        # 超参数
        self.max_iterations = 3  # 减少迭代次数，提高速度
        self.confidence_threshold = 0.25  # 检测器阈值
        self.vlm_threshold = 0.2  # VLM验证阈值（已优化）
    
    def critique_query(self, query, image):
        """
        批判并精炼查询（消除歧义）
        
        参数：
            query: 原始查询字符串
            image: PIL Image对象
            
        返回：
            dict: {"ambiguities": [...], "refined_query": "..."}
        """
        prompt = f"""
分析这个目标检测查询：“{query}"
任务：
1. 识别其中的歧义
2. 提出一个精炼、准确的查询

输出JSON格式：
{{
    "ambiguities": ["不清楚的方面列表"],
    "refined_query": "改进后的查询文本"
}}
"""
        response = self._vlm_query(image, prompt)
        try:
            return json.loads(response)
        except:
            return {"ambiguities": [], "refined_query": query}
    
    def detect(self, image, query, batch_strategy="hierarchical"):
        """
        分批检测（这是核心优化）
        
        参数：
            image: PIL Image
            query: 查询字符串
            batch_strategy: 'single'单批 | 'hierarchical'层次化（推荐）
            
        返回：
            Boxes对象
        """
        if batch_strategy == "single":
            # 单批次检测（适合快速原型）
            results = self.detector(image)
            return results[0].boxes
            
        elif batch_strategy == "hierarchical":
            # 层次化批次（我推荐这个）
            # 批次1：粗分类（阈值放宽，尽量召回）
            results_1 = self.detector(image, conf=0.3)
            boxes_1 = results_1[0].boxes
            
            # 批次2：VLM引导的细分类
            refined_boxes = []
            for i, box in enumerate(boxes_1):
                if box.conf > 0.7:
                    # 高置信度直接保留，不浪费时间
                    refined_boxes.append(box)
                else:
                    # 低置信度的用VLM重新分类
                    crop = self._crop(image, box)
                    true_class = self._vlm_query(crop, f"这是什么物体？用一个词回答。")
                    # 更新类别
                    box.cls_name = true_class.strip().lower()
                    refined_boxes.append(box)
            
            return refined_boxes
    
    def verify_boxes(self, image, boxes, query, strategy="sample"):
        """
        验证边界框
        
        参数：
            image: PIL Image
            boxes: 检测框列表
            query: 查询字符串
            strategy: 'full' 全量 | 'sample' 采样（推荐）
            
        返回：
            List[Box]: 验证通过的框
        """
        if strategy == "full":
            verify_list = boxes
        elif strategy == "sample":
            # 采样策略：前10个高置信度 + 5个可疑框
            boxes_sorted = sorted(boxes, key=lambda x: x.conf, reverse=True)
            high_conf = boxes_sorted[:10]  # 高置信度的
            low_conf = [b for b in boxes if 0.4 < b.conf < 0.6][:5]  # 可疑的
            verify_list = high_conf + low_conf
        
        verified = []
        for box in verify_list:
            crop = self._crop(image, box)
            is_match = self._verify_single_box(crop, query)
            
            if is_match:
                verified.append(box)
        
        return verified
    
    def run(self, image, initial_query):
        """
        运行完整Pipeline（这是总入口）
        
        参数：
            image: PIL Image或图像路径
            initial_query: 初始查询字符串
            
        返回：
            dict: {
                "boxes": List[Box],
                "final_query": str,
                "iterations": int,
                "avg_confidence": float
            }
        """
        # 加载图像
        if isinstance(image, str):
            image = Image.open(image)
            
        query = initial_query
        iteration = 0
        
        print(f"\n{'='*60}")
        print(f"启动Agentic检测pipeline")
        print(f"初始查询: '{initial_query}'")
        print(f"{'='*60}\n")
        
        while iteration < self.max_iterations:
            print(f"--- 第 {iteration + 1} 轮迭代 ---")
            
            # 步骤1: 批判查询 (让VLM挑刺)
            print("→ 批判查询中...")
            critique = self.critique_query(query, image)
            if critique["ambiguities"]:
                print(f"  发现歧义: {critique['ambiguities']}")
                
            # 步骤2: 精炼查询
            query = critique.get("refined_query", query)
            print(f"  精炼后查询: '{query}'")
            
            # 步骤3: 分批检测
            print("→ 检测物体中...")
            boxes = self.detect(image, query, batch_strategy="hierarchical")
            print(f"  检测到 {len(boxes)} 个框")
            
            # 步骤4: 验证边界框
            print("→ VLM验证中...")
            verified_boxes = self.verify_boxes(image, boxes, query, strategy="sample")
            print(f"  验证通过 {len(verified_boxes)} 个框")
            
            # 步骤5: 检查终止条件
            avg_confidence = self._compute_avg_confidence(verified_boxes)
            print(f"  平均置信度: {avg_confidence:.3f}")
            
            # 如果质量足够好，提前终止
            if avg_confidence > 0.8 and len(verified_boxes) > 0:
                break
                
            iteration += 1
        
        print(f"\n{'='*60}")
        print(f"Pipeline完成，共 {iteration + 1} 轮迭代")
        print(f"最终结果：{len(verified_boxes)} 个验证框")
        print(f"{'='*60}\n")
        
        return {
            "boxes": verified_boxes,
            "final_query": query,
            "iterations": iteration + 1,
            "avg_confidence": avg_confidence
        }
    
    # == 下面是辅助方法 ==
    
    def _vlm_query(self, image, prompt):
        """VLM查询的底层实现（封装一下方便用）"""
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        ).to("cuda")
        
        with torch.no_grad():
            outputs = self.vlm.generate(**inputs, max_new_tokens=100)
            
        return self.processor.decode(outputs[0], skip_special_tokens=True)
    
    def _verify_single_box(self, crop, query):
        """单个框的验证"""
        prompt = f"这是一个{query}吗？回答是或否。"
        response = self._vlm_query(crop, prompt)
        return "是" in response.lower() or "yes" in response.lower()
    
    def _crop(self, image, box):
        """裁剪图像区域"""
        coords = box.xyxy[0].cpu().numpy() if hasattr(box, 'xyxy') else box
        return image.crop(coords.tolist())
    
    def _compute_avg_confidence(self, boxes):
        """计算平均置信度"""
        confs = [box.conf for box in boxes]
        return sum(confs) / len(confs) if confs else 0

# == 使用示例 ==
if __name__ == "__main__":
    print("\n" + "="*60)
    print("简化演示 - 使用项目已有的Agent")
    print("="*60 + "\n")
    
    # 初始化Pipeline
    pipeline = AgenticDetectionPipeline()
    
    # 测试图像路径
    test_image = input("请输入图像路径 (直接回车使用默认): ").strip()
    if not test_image:
        test_image = "data/DOTA/images/val/P0005.png"
    
    # 使用项目已有的agent直接检测（不使用迭代式pipeline）
    print(f"\n检测图像: {test_image}")
    
    # 方法1: YOLO基线
    print("\n1. YOLO基线检测...")
    baseline_results = pipeline.agent.baseline_detect(test_image)
    print(f"   → 检测到 {len(baseline_results['boxes'])} 个目标")
    
    # 显示YOLO检测到的类别
    if len(baseline_results['boxes']) > 0:
        from collections import Counter
        class_counts = Counter(baseline_results['class_names'])
        print(f"   → 类别分布: {dict(class_counts)}")
    
    # 方法2: 修正查询格式的Agentic检测
    print("\n2. Agentic检测 - 修正查询格式")
    # ✓ 修正: 使用类别名而不是自然语言指令
    query_fixed = "ship"  # 标准CLIP格式
    print(f"   查询: '{query_fixed}'")
    agentic_results_fixed = pipeline.agent.detect(test_image, query=query_fixed)
    print(f"   → 检测到 {len(agentic_results_fixed['boxes'])} 个目标")
    
    # 方法3: 不使用查询（用类别名验证）
    print("\n3. Agentic检测 - 使用类别名验证")
    agentic_results_no_query = pipeline.agent.detect(test_image, query=None)
    print(f"   → 检测到 {len(agentic_results_no_query['boxes'])} 个目标")
    
    # 方法4: 降低VLM阈值
    print("\n4. Agentic检测 - 降低VLM阈值")
    original_threshold = pipeline.vlm_agent.verification_threshold
    pipeline.vlm_agent.verification_threshold = 0.10
    agentic_results_low_threshold = pipeline.agent.detect(test_image, query="ship")
    pipeline.vlm_agent.verification_threshold = original_threshold  # 恢复
    print(f"   阈值: 0.10")
    print(f"   → 检测到 {len(agentic_results_low_threshold['boxes'])} 个目标")
    
    # 选择最佳结果可视化
    agentic_results = agentic_results_fixed
    if len(agentic_results_no_query['boxes']) > len(agentic_results['boxes']):
        agentic_results = agentic_results_no_query
    
    # 可视化
    print("3. 可视化结果...")
    if len(agentic_results['boxes']) > 0:
        pipeline.visualizer.draw_detections(
            test_image,
            agentic_results,
            show=True
        )
    else:
        print("   ⚠️ 警告: Agentic检测结果为0，可能是VLM阈值过高")
        print("   → 显示YOLO基线结果作为对比...")
        pipeline.visualizer.draw_detections(
            test_image,
            baseline_results,
            show=True
        )
    
    # 分析为什么Agentic结果为0
    if len(agentic_results['boxes']) == 0 and len(baseline_results['boxes']) > 0:
        print("\n" + "="*60)
        print("⚠️ 问题诊断:")
        print("="*60)
        print("YOLO检测到目标，但Agentic方法过滤掉了所有目标")
        print("\n可能原因:")
        print("1. VLM验证阈值过高 (当前: 0.2)")
        print("2. 查询 'find all planes' 与实际类别不匹配")
        print("3. CLIP文本编码与视觉特征相似度低")
        print("\n建议解决方案:")
        print("1. 降低VLM阈值到 0.15")
        print("2. 检查图像中是否真的有planes类别")
        print("3. 尝试更通用的查询，如空字符串（不过滤类别）")
    
    print("\n" + "="*60)
    print("演示完成! ✓")
    print("="*60 + "\n")