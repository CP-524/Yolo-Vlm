#!/usr/bin/env python3
"""VLM验证调试脚本"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.models.model_loader import ModelLoader
from src.agents.vlm_agent import VLMAgent
from src.agents.query_processor import QueryProcessor
from src.utils.data_utils import load_yaml_config
import numpy as np

def debug_vlm_verification():
    """调试VLM验证过程"""
    print("🔍 开始VLM验证调试...")
    
    # 加载配置
    data_config = load_yaml_config('configs/DOTA.yaml')
    class_names = list(data_config['names'].values())
    print(f"📋 DOTA类别: {class_names}")
    
    # 加载模型
    model_loader = ModelLoader('configs/model_configs.yaml')
    yolo_model = model_loader.load_yolo(device='cuda')
    vlm_model = model_loader.load_vlm(device='cuda')
    
    # 创建VLM Agent（使用较低的阈值）
    vlm_agent = VLMAgent(vlm_model, verification_threshold=0.1)  # 降低阈值
    query_processor = QueryProcessor(class_names)
    
    # 测试图像
    image_path = 'data/DOTA/images/val/P0005.png'
    print(f"🖼️ 测试图像: {image_path}")
    
    # YOLO检测
    print("\n1️⃣ YOLO检测...")
    detections = yolo_model.predict(image_path)
    print(f"   YOLO检测到 {len(detections['boxes'])} 个目标")
    
    if len(detections['boxes']) > 0:
        print(f"   前5个检测结果:")
        for i in range(min(5, len(detections['boxes']))):
            print(f"     {detections['class_names'][i]}: {detections['scores'][i]:.3f}")
    
    # 测试不同的VLM阈值
    print("\n2️⃣ 测试不同VLM阈值...")
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    for threshold in thresholds:
        vlm_agent_temp = VLMAgent(vlm_model, verification_threshold=threshold)
        verified_detections = vlm_agent_temp.verify_detections(image_path, detections)
        passed_count = len(verified_detections['boxes'])
        print(f"   阈值 {threshold}: {passed_count}/{len(detections['boxes'])} 通过验证")
    
    # 测试CLIP相似度分数分布
    print("\n3️⃣ 分析CLIP相似度分数...")
    if len(detections['boxes']) > 0:
        # 取前10个检测进行详细分析
        sample_detections = {
            'boxes': detections['boxes'][:10],
            'scores': detections['scores'][:10],
            'classes': detections['classes'][:10],
            'class_names': detections['class_names'][:10]
        }
        
        # 手动计算相似度分数
        from PIL import Image
        image = Image.open(image_path)
        
        # 裁剪检测框
        cropped_images = []
        for box in sample_detections['boxes']:
            x1, y1, x2, y2 = map(int, box)
            cropped = image.crop((x1, y1, x2, y2))
            cropped_images.append(cropped)
        
        # 计算相似度
        similarities = []
        for i, (cropped_img, class_name) in enumerate(zip(cropped_images, sample_detections['class_names'])):
            query = f"a photo of a {class_name}"
            similarity = vlm_model.compute_similarity(cropped_img, query)[0, 0]
            similarities.append(similarity)
            print(f"   {class_name}: {similarity:.4f}")
        
        print(f"\n📊 相似度统计:")
        print(f"   平均: {np.mean(similarities):.4f}")
        print(f"   中位数: {np.median(similarities):.4f}")
        print(f"   最小值: {np.min(similarities):.4f}")
        print(f"   最大值: {np.max(similarities):.4f}")
    
    # 推荐阈值
    print("\n4️⃣ 推荐设置...")
    print("   💡 建议VLM验证阈值: 0.2-0.3")
    print("   💡 或者禁用VLM验证进行测试")

if __name__ == "__main__":
    debug_vlm_verification()
