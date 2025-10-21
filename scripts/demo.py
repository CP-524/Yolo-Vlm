"""演示脚本 - 单张图像检测"""
import argparse
import logging
from pathlib import Path

import sys 
sys.path.append(str(Path(__file__).parent.parent))
from src.models.model_loader import ModelLoader
from src.agents.detection_agent import DetectionAgent
from src.agents.vlm_agent import VLMAgent
from src.agents.query_processor import QueryProcessor
from src.utils.visualization import Visualizer
from src.utils.data_utils import load_yaml_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Demo detection on a single image')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--weights', type=str, default=None,
                       help='Path to YOLO weights (overrides config file)')
    parser.add_argument('--model-config', type=str, default='configs/model_configs.yaml',
                       help='Path to model config file')
    parser.add_argument('--pipeline-config', type=str, default='configs/pipeline_configs.yaml',
                       help='Path to pipeline config file')
    parser.add_argument('--data-config', type=str, default='configs/DOTA.yaml',
                       help='Path to dataset config file')
    parser.add_argument('--query', type=str, default=None,
                       help='Optional natural language query')
    parser.add_argument('--method', type=str, default='agent',
                       choices=['yolo', 'agent', 'compare'],
                       help='Detection method')
    parser.add_argument('--output', type=str, default=None,
                       help='Output image path')
    parser.add_argument('--show', action='store_true',
                       help='Display result')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')
    return parser.parse_args()


def main():
    args = parse_args()
    
    logger.info(f"Running demo on: {args.image}")
    
    # 加载配置
    model_config = load_yaml_config(args.model_config)
    pipeline_config = load_yaml_config(args.pipeline_config)
    data_config = load_yaml_config(args.data_config)
    
    # 获取类别名称
    class_names = list(data_config['names'].values())
    
    # 加载模型
    logger.info("Loading models...")
    model_loader = ModelLoader(args.model_config)
    
    # 如果指定了weights参数，使用它；否则使用配置文件中的路径
    yolo_weights = args.weights if args.weights else None
    if yolo_weights:
        logger.info(f"Using custom weights: {yolo_weights}")
    
    yolo_model = model_loader.load_yolo(model_path=yolo_weights, device=args.device)
    
    # 创建Agent
    enable_vlm = pipeline_config['agentic_pipeline'].get('enable_vlm', True) and args.method != 'yolo'
    vlm_agent = None
    
    if enable_vlm:
        vlm_model = model_loader.load_vlm(device=args.device)
        vlm_agent = VLMAgent(
            vlm_model,
            verification_threshold=pipeline_config['vlm_verification'].get('verification_threshold', 0.3)
        )
    
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
    if args.method == 'compare':
        logger.info("Running comparison...")
        results = detection_agent.compare_methods(args.image, args.query)
        
        # 可视化对比
        output_path = args.output or f"outputs/visualizations/compare_{Path(args.image).stem}.jpg"
        visualizer.compare_results(
            args.image,
            results['baseline'],
            results['agentic'],
            output_path=output_path,
            show=args.show
        )
        
        # 打印结果
        print(f"\nYOLO detections: {len(results['baseline']['boxes'])}")
        print(f"Agent detections: {len(results['agentic']['boxes'])}")
        
    elif args.method == 'yolo':
        logger.info("Running YOLO detection...")
        results = detection_agent.baseline_detect(args.image)
        
        # 可视化
        output_path = args.output or f"outputs/visualizations/yolo_{Path(args.image).stem}.jpg"
        visualizer.draw_detections(
            args.image,
            results,
            output_path=output_path,
            show=args.show
        )
        
        print(f"\nDetections: {len(results['boxes'])}")
        
    else:  # agent
        logger.info("Running agent detection...")
        if args.query:
            logger.info(f"Query: {args.query}")
        
        results = detection_agent.detect(args.image, query=args.query)
        
        # 可视化
        output_path = args.output or f"outputs/visualizations/agent_{Path(args.image).stem}.jpg"
        visualizer.draw_detections(
            args.image,
            results,
            output_path=output_path,
            show=args.show
        )
        
        print(f"\nDetections: {len(results['boxes'])}")
        if 'class_names' in results:
            for i, name in enumerate(results['class_names']):
                score = results['scores'][i]
                print(f"  {name}: {score:.3f}")
    
    logger.info("Demo completed!")


if __name__ == '__main__':
    main()
