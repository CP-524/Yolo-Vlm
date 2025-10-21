#!/usr/bin/env python3
"""交互式检测脚本 - 直接在程序中配置参数"""
import sys
from pathlib import Path
import logging

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from src.models.model_loader import ModelLoader
from src.agents.detection_agent import DetectionAgent
from src.agents.vlm_agent import VLMAgent
from src.agents.query_processor import QueryProcessor
from src.utils.visualization import Visualizer
from src.utils.data_utils import load_yaml_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class InteractiveDetector:
    """交互式检测器类"""
    
    def __init__(self):
        """初始化检测器"""
        self.config = self._load_configs()
        self.class_names = list(self.config['data']['names'].values())
        self.detection_agent = None
        self.visualizer = None
        
    def _load_configs(self):
        """加载配置文件"""
        model_config = load_yaml_config('configs/model_configs.yaml')
        pipeline_config = load_yaml_config('configs/pipeline_configs.yaml')
        data_config = load_yaml_config('configs/DOTA.yaml')
        
        return {
            'model': model_config,
            'pipeline': pipeline_config,
            'data': data_config
        }
    
    def setup_models(self, device='cuda', yolo_weights=None):
        """设置模型"""
        logger.info("Loading models...")
        
        # 加载模型
        model_loader = ModelLoader('configs/model_configs.yaml')
        yolo_model = model_loader.load_yolo(model_path=yolo_weights, device=device)
        
        # 创建VLM Agent（如果需要）
        enable_vlm = self.config['pipeline']['agentic_pipeline'].get('enable_vlm', True)
        vlm_agent = None
        
        if enable_vlm:
            try:
                vlm_model = model_loader.load_vlm(device=device)
                vlm_agent = VLMAgent(
                    vlm_model,
                    verification_threshold=self.config['pipeline']['vlm_verification'].get('verification_threshold', 0.3)
                )
            except Exception as e:
                logger.warning(f"Failed to load VLM model: {e}")
                logger.warning("Continuing without VLM verification...")
                enable_vlm = False
        
        # 创建查询处理器
        query_processor = QueryProcessor(self.class_names)
        
        # 创建检测Agent
        self.detection_agent = DetectionAgent(
            yolo_model=yolo_model,
            vlm_agent=vlm_agent,
            query_processor=query_processor,
            enable_vlm=enable_vlm
        )
        
        # 创建可视化器
        self.visualizer = Visualizer(self.class_names)
        
        logger.info("Models loaded successfully!")
    
    def detect_single_image(self, image_path, method='agent', query=None, show=True, save_output=True):
        """检测单张图像"""
        if not self.detection_agent:
            logger.error("Models not loaded! Please call setup_models() first.")
            return None
        
        logger.info(f"Running {method} detection on: {image_path}")
        
        # 执行检测
        if method == 'compare':
            logger.info("Running comparison...")
            results = self.detection_agent.compare_methods(image_path, query)
            
            # 可视化对比
            output_path = f"outputs/visualizations/compare_{Path(image_path).stem}.jpg"
            self.visualizer.compare_results(
                image_path,
                results['baseline'],
                results['agentic'],
                output_path=output_path if save_output else None,
                show=show
            )
            
            # 打印结果
            print(f"\nYOLO detections: {len(results['baseline']['boxes'])}")
            print(f"Agent detections: {len(results['agentic']['boxes'])}")
            
            return results
            
        elif method == 'yolo':
            logger.info("Running YOLO detection...")
            results = self.detection_agent.baseline_detect(image_path)
            
            # 可视化
            output_path = f"outputs/visualizations/yolo_{Path(image_path).stem}.jpg"
            self.visualizer.draw_detections(
                image_path,
                results,
                output_path=output_path if save_output else None,
                show=show
            )
            
            print(f"\nDetections: {len(results['boxes'])}")
            
            return results
            
        else:  # agent
            logger.info("Running agent detection...")
            if query:
                logger.info(f"Query: {query}")
            
            results = self.detection_agent.detect(image_path, query=query)
            
            # 可视化
            output_path = f"outputs/visualizations/agent_{Path(image_path).stem}.jpg"
            self.visualizer.draw_detections(
                image_path,
                results,
                output_path=output_path if save_output else None,
                show=show
            )
            
            print(f"\nDetections: {len(results['boxes'])}")
            if 'class_names' in results and len(results['boxes']) > 0:
                print("Detected objects:")
                for i, name in enumerate(results['class_names']):
                    score = results['scores'][i]
                    print(f"  {name}: {score:.3f}")
            
            return results
    
    def batch_detect(self, image_paths, method='agent', queries=None, show=False, save_output=True):
        """批量检测"""
        if not self.detection_agent:
            logger.error("Models not loaded! Please call setup_models() first.")
            return None
        
        logger.info(f"Running batch {method} detection on {len(image_paths)} images")
        
        results = []
        for i, image_path in enumerate(image_paths):
            query = queries[i] if queries and i < len(queries) else None
            result = self.detect_single_image(
                image_path, 
                method=method, 
                query=query, 
                show=show, 
                save_output=save_output
            )
            results.append(result)
        
        return results
    
    def get_available_images(self, data_dir='data/DOTA/images/val'):
        """获取可用的图像列表"""
        image_dir = Path(data_dir)
        if not image_dir.exists():
            logger.warning(f"Directory {data_dir} does not exist")
            return []
        
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(image_dir.glob(f'*{ext}'))
            image_files.extend(image_dir.glob(f'*{ext.upper()}'))
        
        return sorted([str(f) for f in image_files])
    
    def print_class_names(self):
        """打印可用的类别名称"""
        print("\nAvailable DOTA classes:")
        for i, name in enumerate(self.class_names):
            print(f"  {i}: {name}")


def main():
    """主函数 - 演示如何使用交互式检测器"""
    
    # ==================== 配置参数 ====================
    # 在这里直接修改参数，不需要命令行输入
    
    # 基本配置
    IMAGE_PATH = "data/DOTA/images/val/P0005.png"  # 图像路径
    METHOD = "agent"  # 检测方法: 'yolo', 'agent', 'compare'
    QUERY = "find all planes"  # 查询文本（可选）
    DEVICE = "cuda"  # 设备: 'cuda' 或 'cpu'
    YOLO_WEIGHTS = None  # YOLO权重路径（可选，None使用默认）
    
    # 显示和保存配置
    SHOW_RESULT = True  # 是否显示结果
    SAVE_OUTPUT = True  # 是否保存输出图像
    
    # ==================== 执行检测 ====================
    
    print("🚀 Interactive YOLO Agent Detection")
    print("=" * 50)
    
    # 创建检测器
    detector = InteractiveDetector()
    
    # 打印可用类别
    detector.print_class_names()
    
    # 设置模型
    detector.setup_models(device=DEVICE, yolo_weights=YOLO_WEIGHTS)
    
    # 检查图像是否存在
    if not Path(IMAGE_PATH).exists():
        print(f"\n❌ Image not found: {IMAGE_PATH}")
        print("Available images:")
        available_images = detector.get_available_images()
        for img in available_images[:5]:  # 显示前5个
            print(f"  {img}")
        if len(available_images) > 5:
            print(f"  ... and {len(available_images) - 5} more")
        return
    
    # 执行检测
    print(f"\n🔍 Detection Configuration:")
    print(f"  Image: {IMAGE_PATH}")
    print(f"  Method: {METHOD}")
    print(f"  Query: {QUERY if QUERY else 'None'}")
    print(f"  Device: {DEVICE}")
    print(f"  Show: {SHOW_RESULT}")
    print(f"  Save: {SAVE_OUTPUT}")
    
    try:
        results = detector.detect_single_image(
            image_path=IMAGE_PATH,
            method=METHOD,
            query=QUERY,
            show=SHOW_RESULT,
            save_output=SAVE_OUTPUT
        )
        
        if results:
            print("\n✅ Detection completed successfully!")
        else:
            print("\n❌ Detection failed!")
            
    except Exception as e:
        logger.error(f"Detection failed: {e}")
        print(f"\n❌ Error: {e}")


def demo_batch_detection():
    """演示批量检测"""
    
    # ==================== 批量检测配置 ====================
    
    # 图像列表
    IMAGE_PATHS = [
        "data/DOTA/images/val/P0003.png",
        "data/DOTA/images/val/P0005.png",
        "data/DOTA/images/val/P0007.png"
    ]
    
    # 对应的查询（可选）
    QUERIES = [
        "find all planes",
        "find all ships", 
        "find all vehicles"
    ]
    
    METHOD = "agent"
    DEVICE = "cuda"
    SHOW_RESULT = False  # 批量检测时不显示
    SAVE_OUTPUT = True
    
    # ==================== 执行批量检测 ====================
    
    print("🚀 Batch Detection Demo")
    print("=" * 50)
    
    detector = InteractiveDetector()
    detector.setup_models(device=DEVICE)
    
    # 过滤存在的图像
    existing_images = []
    existing_queries = []
    
    for i, img_path in enumerate(IMAGE_PATHS):
        if Path(img_path).exists():
            existing_images.append(img_path)
            if i < len(QUERIES):
                existing_queries.append(QUERIES[i])
            else:
                existing_queries.append(None)
        else:
            print(f"⚠️ Image not found: {img_path}")
    
    if not existing_images:
        print("❌ No valid images found!")
        return
    
    print(f"📸 Processing {len(existing_images)} images...")
    
    try:
        results = detector.batch_detect(
            image_paths=existing_images,
            method=METHOD,
            queries=existing_queries,
            show=SHOW_RESULT,
            save_output=SAVE_OUTPUT
        )
        
        print("\n✅ Batch detection completed!")
        
        # 打印汇总结果
        total_detections = 0
        for i, result in enumerate(results):
            if result:
                count = len(result['boxes'])
                total_detections += count
                print(f"  {existing_images[i]}: {count} detections")
        
        print(f"\nTotal detections: {total_detections}")
        
    except Exception as e:
        logger.error(f"Batch detection failed: {e}")
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    # 运行单张图像检测
    main()
    
    # 取消注释下面的行来运行批量检测演示
    # demo_batch_detection()
