"""评估脚本"""
import argparse
import logging
from pathlib import Path
import yaml
import sys 
sys.path.append(str(Path(__file__).parent.parent))
from src.models.model_loader import ModelLoader
from src.agents.detection_agent import DetectionAgent
from src.agents.vlm_agent import VLMAgent
from src.agents.query_processor import QueryProcessor
from src.evaluation.dota_evaluator import DOTAEvaluator
from src.utils.data_utils import load_yaml_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate detection on DOTA dataset')
    parser.add_argument('--model-config', type=str, default='configs/model_configs.yaml',
                       help='Path to model config file')
    parser.add_argument('--pipeline-config', type=str, default='configs/pipeline_configs.yaml',
                       help='Path to pipeline config file')
    parser.add_argument('--data-config', type=str, default='configs/DOTA.yaml',
                       help='Path to dataset config file')
    parser.add_argument('--data-root', type=str, default='data/DOTA',
                       help='Data root directory')
    parser.add_argument('--split', type=str, default='val',
                       help='Dataset split (train/val/test)')
    parser.add_argument('--method', type=str, default='compare',
                       choices=['yolo', 'agent', 'compare'],
                       help='Evaluation method')
    parser.add_argument('--query', type=str, default=None,
                       help='Optional query for agentic method')
    parser.add_argument('--max-images', type=int, default=None,
                       help='Maximum number of images to evaluate')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')
    return parser.parse_args()


def main():
    args = parse_args()
    
    logger.info("Starting evaluation")
    
    # 加载配置
    model_config = load_yaml_config(args.model_config)
    pipeline_config = load_yaml_config(args.pipeline_config)
    data_config = load_yaml_config(args.data_config)
    
    # 获取类别名称
    class_names = list(data_config['names'].values())
    
    # 加载模型
    logger.info("Loading models...")
    model_loader = ModelLoader(args.model_config)
    yolo_model = model_loader.load_yolo(device=args.device)
    
    # 创建Agent
    enable_vlm = pipeline_config['agentic_pipeline'].get('enable_vlm', True)
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
    
    # 创建评估器
    evaluator = DOTAEvaluator(
        detection_agent=detection_agent,
        data_root=args.data_root,
        class_names=class_names,
        split=args.split
    )
    
    # 执行评估
    if args.method == 'compare':
        results = evaluator.compare_methods(
            query=args.query,
            max_images=args.max_images
        )
    elif args.method == 'yolo':
        results = evaluator.evaluate(
            use_agentic=False,
            max_images=args.max_images
        )
    else:  # agent
        results = evaluator.evaluate(
            use_agentic=True,
            query=args.query,
            max_images=args.max_images
        )
    
    logger.info("Evaluation completed!")


if __name__ == '__main__':
    main()
