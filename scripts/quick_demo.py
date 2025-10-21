#!/usr/bin/env python3
"""快速配置检测脚本 - 直接在代码中修改参数"""
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

# ==================== 在这里修改你的配置 ====================

# 基本配置
IMAGE_PATH = "data/DOTA/images/val/P0005.png"  # 要检测的图像路径
METHOD = "agent"  # 检测方法: 'yolo', 'agent', 'compare'
QUERY = "find all planes"  # 查询文本（可选，None表示无查询）
DEVICE = "cuda"  # 设备: 'cuda' 或 'cpu'

# 显示配置
SHOW_RESULT = True  # 是否显示检测结果
SAVE_RESULT = True  # 是否保存检测结果图像

# ==================== 配置结束，下面是执行代码 ====================

# QUERY = "find all planes"           # 查找所有飞机
# QUERY = "find all ships"            # 查找所有船只
# QUERY = "find all helicopters"     # 查找所有直升机
# QUERY = "find all bridges"         # 查找所有桥梁
# QUERY = "find all vehicles"        # 查找所有车辆
# QUERY = "find at least 3 planes"    # 查找至少3架飞机
# QUERY = "find exactly 2 ships"     # 查找恰好2艘船
# QUERY = "find more than 5 vehicles" # 查找超过5辆车
# QUERY = "find all planes"  
# QUERY = "find all aircraft"         # 查找所有飞行器（飞机+直升机）
# QUERY = "find all transportation"   # 查找所有交通工具
# QUERY = "find all infrastructure"   # 查找所有基础设施
# QUERY = "find all sports facilities" # 查找所有体育设施
# QUERY = "find large objects"        # 查找大型物体
# QUERY = "find flying objects"       # 查找飞行物体
# QUERY = "find water vehicles"       # 查找水上交通工具
# QUERY = "find ground vehicles"      # 查找地面交通工具
# QUERY = "find all objects in harbor" # 查找港口中的所有物体
# QUERY = "find military aircraft"    # 查找军用飞机
# QUERY = "find commercial vehicles"   # 查找商用车辆
# QUERY = "find recreational facilities" # 查找娱乐设施

def main():
    """主函数"""
    print("🚀 Quick Detection Script")
    print("=" * 40)
    print(f"Image: {IMAGE_PATH}")
    print(f"Method: {METHOD}")
    print(f"Query: {QUERY if QUERY else 'None'}")
    print(f"Device: {DEVICE}")
    print("=" * 40)
    
    try:
        # 导入必要的模块
        from src.models.model_loader import ModelLoader
        from src.agents.detection_agent import DetectionAgent
        from src.agents.vlm_agent import VLMAgent
        from src.agents.query_processor import QueryProcessor
        from src.utils.visualization import Visualizer
        from src.utils.data_utils import load_yaml_config
        import logging
        
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logger = logging.getLogger(__name__)
        
        # 检查图像是否存在
        if not Path(IMAGE_PATH).exists():
            print(f"❌ Image not found: {IMAGE_PATH}")
            print("Available images in data/DOTA/images/val/:")
            val_dir = Path("data/DOTA/images/val")
            if val_dir.exists():
                images = list(val_dir.glob("*.png"))[:5]
                for img in images:
                    print(f"  {img}")
            return
        
        # 加载配置
        logger.info("Loading configurations...")
        model_config = load_yaml_config('configs/model_configs.yaml')
        pipeline_config = load_yaml_config('configs/pipeline_configs.yaml')
        data_config = load_yaml_config('configs/DOTA.yaml')
        
        # 获取类别名称
        class_names = list(data_config['names'].values())
        print(f"\n📋 Available classes: {', '.join(class_names)}")
        
        # 加载模型
        logger.info("Loading models...")
        model_loader = ModelLoader('configs/model_configs.yaml')
        yolo_model = model_loader.load_yolo(device=DEVICE)
        
        # 创建Agent组件
        enable_vlm = pipeline_config['agentic_pipeline'].get('enable_vlm', True) and METHOD != 'yolo'
        vlm_agent = None
        
        if enable_vlm:
            try:
                vlm_model = model_loader.load_vlm(device=DEVICE)
                vlm_agent = VLMAgent(
                    vlm_model,
                    verification_threshold=pipeline_config['vlm_verification'].get('verification_threshold', 0.3)
                )
                logger.info("VLM model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load VLM model: {e}")
                logger.warning("Continuing without VLM verification...")
                enable_vlm = False
        
        # 创建查询处理器
        query_processor = QueryProcessor(class_names)
        
        # 创建检测Agent
        detection_agent = DetectionAgent(
            yolo_model=yolo_model,
            vlm_agent=vlm_agent,
            query_processor=query_processor,
            enable_vlm=enable_vlm
        )
        
        # 创建可视化器
        visualizer = Visualizer(class_names)
        
        # 执行检测
        logger.info(f"Running {METHOD} detection...")
        
        if METHOD == 'compare':
            logger.info("Running comparison...")
            results = detection_agent.compare_methods(IMAGE_PATH, QUERY)
            
            # 可视化对比
            output_path = f"outputs/visualizations/compare_{Path(IMAGE_PATH).stem}.jpg"
            visualizer.compare_results(
                IMAGE_PATH,
                results['baseline'],
                results['agentic'],
                output_path=output_path if SAVE_RESULT else None,
                show=SHOW_RESULT
            )
            
            # 打印结果
            print(f"\n📊 Results:")
            print(f"  YOLO detections: {len(results['baseline']['boxes'])}")
            print(f"  Agent detections: {len(results['agentic']['boxes'])}")
            
        elif METHOD == 'yolo':
            logger.info("Running YOLO detection...")
            results = detection_agent.baseline_detect(IMAGE_PATH)
            
            # 可视化
            output_path = f"outputs/visualizations/yolo_{Path(IMAGE_PATH).stem}.jpg"
            visualizer.draw_detections(
                IMAGE_PATH,
                results,
                output_path=output_path if SAVE_RESULT else None,
                show=SHOW_RESULT
            )
            
            print(f"\n📊 Results:")
            print(f"  Detections: {len(results['boxes'])}")
            
        else:  # agent
            logger.info("Running agent detection...")
            if QUERY:
                logger.info(f"Query: {QUERY}")
            
            results = detection_agent.detect(IMAGE_PATH, query=QUERY)
            
            # 可视化
            output_path = f"outputs/visualizations/agent_{Path(IMAGE_PATH).stem}.jpg"
            visualizer.draw_detections(
                IMAGE_PATH,
                results,
                output_path=output_path if SAVE_RESULT else None,
                show=SHOW_RESULT
            )
            
            print(f"\n📊 Results:")
            print(f"  Detections: {len(results['boxes'])}")
            
            if 'class_names' in results and len(results['boxes']) > 0:
                print("  Detected objects:")
                for i, name in enumerate(results['class_names']):
                    score = results['scores'][i]
                    print(f"    {name}: {score:.3f}")
        
        print("\n✅ Detection completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
