"""优化脚本 - 超参数搜索"""
import argparse
import logging
from pathlib import Path
import itertools
import json

from src.models.model_loader import ModelLoader
from src.agents.detection_agent import DetectionAgent
from src.agents.vlm_agent import VLMAgent
from src.agents.query_processor import QueryProcessor
from src.evaluation.dota_evaluator import DOTAEvaluator
from src.utils.data_utils import load_yaml_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Hyperparameter optimization')
    parser.add_argument('--model-config', type=str, default='configs/model_configs.yaml',
                       help='Path to model config file')
    parser.add_argument('--pipeline-config', type=str, default='configs/pipeline_configs.yaml',
                       help='Path to pipeline config file')
    parser.add_argument('--data-config', type=str, default='configs/DOTA.yaml',
                       help='Path to dataset config file')
    parser.add_argument('--data-root', type=str, default='data/DOTA',
                       help='Data root directory')
    parser.add_argument('--max-images', type=int, default=100,
                       help='Maximum number of images for optimization')
    parser.add_argument('--output', type=str, default='outputs/optimization_results.json',
                       help='Output path for results')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')
    return parser.parse_args()


def main():
    args = parse_args()
    
    logger.info("Starting hyperparameter optimization")
    
    # 加载配置
    model_config = load_yaml_config(args.model_config)
    pipeline_config = load_yaml_config(args.pipeline_config)
    data_config = load_yaml_config(args.data_config)
    
    # 获取类别名称
    class_names = list(data_config['names'].values())
    
    # 定义搜索空间
    conf_thresholds = [0.2, 0.25, 0.3]
    vlm_thresholds = [0.25, 0.3, 0.35, 0.4]
    
    # 加载YOLO模型(只需加载一次)
    logger.info("Loading YOLO model...")
    model_loader = ModelLoader(args.model_config)
    yolo_model = model_loader.load_yolo(device=args.device)
    
    # 加载VLM模型
    logger.info("Loading VLM model...")
    vlm_model = model_loader.load_vlm(device=args.device)
    
    # 存储结果
    all_results = []
    
    # 网格搜索
    total_combinations = len(conf_thresholds) * len(vlm_thresholds)
    logger.info(f"Testing {total_combinations} hyperparameter combinations")
    
    for i, (conf_th, vlm_th) in enumerate(itertools.product(conf_thresholds, vlm_thresholds)):
        logger.info(f"\n[{i+1}/{total_combinations}] Testing conf={conf_th}, vlm={vlm_th}")
        
        # 更新YOLO置信度阈值
        yolo_model.conf_threshold = conf_th
        
        # 创建VLM Agent
        vlm_agent = VLMAgent(
            vlm_model,
            verification_threshold=vlm_th
        )
        
        # 创建查询处理器
        query_processor = QueryProcessor(class_names)
        
        # 创建检测Agent
        detection_agent = DetectionAgent(
            yolo_model=yolo_model,
            vlm_agent=vlm_agent,
            query_processor=query_processor,
            enable_vlm=True
        )
        
        # 创建评估器
        evaluator = DOTAEvaluator(
            detection_agent=detection_agent,
            data_root=args.data_root,
            class_names=class_names,
            split='val'
        )
        
        # 评估
        metrics = evaluator.evaluate(
            use_agentic=True,
            max_images=args.max_images
        )
        
        # 保存结果
        result = {
            'conf_threshold': conf_th,
            'vlm_threshold': vlm_th,
            'map': metrics['map'],
            'mean_precision': metrics['mean_precision'],
            'mean_recall': metrics['mean_recall'],
            'mean_f1': metrics['mean_f1']
        }
        all_results.append(result)
        
        logger.info(f"Results: mAP={metrics['map']:.4f}, P={metrics['mean_precision']:.4f}, "
                   f"R={metrics['mean_recall']:.4f}, F1={metrics['mean_f1']:.4f}")
    
    # 找到最佳配置
    best_result = max(all_results, key=lambda x: x['map'])
    
    logger.info("\n" + "="*80)
    logger.info("OPTIMIZATION RESULTS")
    logger.info("="*80)
    logger.info(f"Best configuration:")
    logger.info(f"  Confidence threshold: {best_result['conf_threshold']}")
    logger.info(f"  VLM threshold: {best_result['vlm_threshold']}")
    logger.info(f"  mAP: {best_result['map']:.4f}")
    logger.info(f"  Precision: {best_result['mean_precision']:.4f}")
    logger.info(f"  Recall: {best_result['mean_recall']:.4f}")
    logger.info(f"  F1: {best_result['mean_f1']:.4f}")
    logger.info("="*80)
    
    # 保存所有结果
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump({
            'best': best_result,
            'all_results': all_results
        }, f, indent=2)
    
    logger.info(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
