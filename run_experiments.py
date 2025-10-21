"""实验运行入口"""
import argparse
import logging
from pathlib import Path
import json
from datetime import datetime

from src.models.model_loader import ModelLoader
from src.agents.detection_agent import DetectionAgent
from src.agents.vlm_agent import VLMAgent
from src.agents.query_processor import QueryProcessor
from src.evaluation.dota_evaluator import DOTAEvaluator
from src.evaluation.benchmark import Benchmark
from src.utils.data_utils import load_yaml_config
from src.utils.visualization import Visualizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Run experiments on YOLO Agent')
    parser.add_argument('--experiment', type=str, default='evaluate',
                       choices=['evaluate', 'benchmark', 'ablation'],
                       help='Experiment type')
    parser.add_argument('--model-config', type=str, default='configs/model_configs.yaml',
                       help='Model configuration')
    parser.add_argument('--pipeline-config', type=str, default='configs/pipeline_configs.yaml',
                       help='Pipeline configuration')
    parser.add_argument('--data-config', type=str, default='configs/DOTA.yaml',
                       help='Dataset configuration')
    parser.add_argument('--data-root', type=str, default='data/DOTA',
                       help='Data root directory')
    parser.add_argument('--output-dir', type=str, default='experiments',
                       help='Output directory')
    parser.add_argument('--max-images', type=int, default=None,
                       help='Maximum images to process')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')
    return parser.parse_args()


def run_evaluation(args):
    """运行评估实验"""
    logger.info("=== Running Evaluation Experiment ===")
    
    # 加载配置
    model_config = load_yaml_config(args.model_config)
    pipeline_config = load_yaml_config(args.pipeline_config)
    data_config = load_yaml_config(args.data_config)
    class_names = list(data_config['names'].values())
    
    # 加载模型
    model_loader = ModelLoader(args.model_config)
    yolo_model = model_loader.load_yolo(device=args.device)
    vlm_model = model_loader.load_vlm(device=args.device)
    
    # 创建Agent
    vlm_agent = VLMAgent(vlm_model, verification_threshold=0.3)
    query_processor = QueryProcessor(class_names)
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
    
    # 执行评估
    results = evaluator.compare_methods(max_images=args.max_images)
    
    # 保存结果
    output_dir = Path(args.output_dir) / 'evaluation'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f'results_{timestamp}.json'
    
    with open(output_file, 'w') as f:
        # 转换numpy类型
        results_serializable = {
            'baseline': {
                'map': float(results['baseline']['map']),
                'mean_precision': float(results['baseline']['mean_precision']),
                'mean_recall': float(results['baseline']['mean_recall']),
                'mean_f1': float(results['baseline']['mean_f1'])
            },
            'agentic': {
                'map': float(results['agentic']['map']),
                'mean_precision': float(results['agentic']['mean_precision']),
                'mean_recall': float(results['agentic']['mean_recall']),
                'mean_f1': float(results['agentic']['mean_f1'])
            },
            'improvement': results['improvement']
        }
        json.dump(results_serializable, f, indent=2)
    
    logger.info(f"Results saved to {output_file}")
    return results


def run_benchmark(args):
    """运行性能基准测试"""
    logger.info("=== Running Benchmark Experiment ===")
    
    # 加载配置
    model_config = load_yaml_config(args.model_config)
    data_config = load_yaml_config(args.data_config)
    class_names = list(data_config['names'].values())
    
    # 加载模型
    model_loader = ModelLoader(args.model_config)
    yolo_model = model_loader.load_yolo(device=args.device)
    vlm_model = model_loader.load_vlm(device=args.device)
    
    # 创建Agent
    vlm_agent = VLMAgent(vlm_model)
    query_processor = QueryProcessor(class_names)
    detection_agent = DetectionAgent(
        yolo_model=yolo_model,
        vlm_agent=vlm_agent,
        query_processor=query_processor,
        enable_vlm=True
    )
    
    # 获取测试图像
    from src.utils.data_utils import get_image_list
    image_dir = Path(args.data_root) / 'images' / 'val'
    image_files = get_image_list(image_dir)
    
    if args.max_images:
        image_files = image_files[:args.max_images]
    
    image_paths = [str(img) for img in image_files]
    
    # 创建基准测试
    benchmark = Benchmark()
    
    # 测试基线方法
    logger.info("Benchmarking baseline method...")
    baseline_results = benchmark.benchmark_detection(
        detection_agent,
        image_paths,
        use_agentic=False
    )
    
    # 测试Agentic方法
    logger.info("Benchmarking agentic method...")
    agentic_results = benchmark.benchmark_detection(
        detection_agent,
        image_paths,
        use_agentic=True
    )
    
    # 打印结果
    print("\n" + "="*80)
    print("BENCHMARK RESULTS")
    print("="*80)
    print(f"\nBaseline Method:")
    print(f"  Mean Latency: {baseline_results['mean_latency']:.4f}s")
    print(f"  FPS: {baseline_results['fps']:.2f}")
    print(f"\nAgentic Method:")
    print(f"  Mean Latency: {agentic_results['mean_latency']:.4f}s")
    print(f"  FPS: {agentic_results['fps']:.2f}")
    print(f"\nOverhead:")
    overhead = agentic_results['mean_latency'] - baseline_results['mean_latency']
    print(f"  Additional Latency: {overhead:.4f}s")
    print(f"  Relative Overhead: {(overhead/baseline_results['mean_latency']*100):.1f}%")
    print("="*80 + "\n")
    
    # 保存结果
    output_dir = Path(args.output_dir) / 'benchmark'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f'benchmark_{timestamp}.json'
    
    with open(output_file, 'w') as f:
        json.dump({
            'baseline': baseline_results,
            'agentic': agentic_results
        }, f, indent=2)
    
    logger.info(f"Results saved to {output_file}")
    return {'baseline': baseline_results, 'agentic': agentic_results}


def run_ablation(args):
    """运行消融实验"""
    logger.info("=== Running Ablation Study ===")
    
    # 加载配置
    model_config = load_yaml_config(args.model_config)
    data_config = load_yaml_config(args.data_config)
    class_names = list(data_config['names'].values())
    
    # 加载模型
    model_loader = ModelLoader(args.model_config)
    yolo_model = model_loader.load_yolo(device=args.device)
    vlm_model = model_loader.load_vlm(device=args.device)
    
    vlm_agent = VLMAgent(vlm_model)
    query_processor = QueryProcessor(class_names)
    
    # 测试不同配置
    configs = [
        {'name': 'Baseline', 'enable_vlm': False, 'enable_query': False},
        {'name': 'YOLO + Query', 'enable_vlm': False, 'enable_query': True},
        {'name': 'YOLO + VLM', 'enable_vlm': True, 'enable_query': False},
        {'name': 'Full Agentic', 'enable_vlm': True, 'enable_query': True}
    ]
    
    results = {}
    
    for config in configs:
        logger.info(f"\nTesting configuration: {config['name']}")
        
        # 创建Agent
        detection_agent = DetectionAgent(
            yolo_model=yolo_model,
            vlm_agent=vlm_agent if config['enable_vlm'] else None,
            query_processor=query_processor if config['enable_query'] else None,
            enable_vlm=config['enable_vlm'],
            enable_query_processing=config['enable_query']
        )
        
        # 评估
        evaluator = DOTAEvaluator(
            detection_agent=detection_agent,
            data_root=args.data_root,
            class_names=class_names,
            split='val'
        )
        
        metrics = evaluator.evaluate(
            use_agentic=config['enable_vlm'] or config['enable_query'],
            max_images=args.max_images
        )
        
        results[config['name']] = {
            'map': float(metrics['map']),
            'precision': float(metrics['mean_precision']),
            'recall': float(metrics['mean_recall']),
            'f1': float(metrics['mean_f1'])
        }
    
    # 打印对比结果
    print("\n" + "="*80)
    print("ABLATION STUDY RESULTS")
    print("="*80)
    print(f"{'Configuration':<20} {'mAP':>8} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print("-"*80)
    for name, metrics in results.items():
        print(f"{name:<20} {metrics['map']:>8.4f} {metrics['precision']:>10.4f} "
              f"{metrics['recall']:>8.4f} {metrics['f1']:>8.4f}")
    print("="*80 + "\n")
    
    # 保存结果
    output_dir = Path(args.output_dir) / 'ablation'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f'ablation_{timestamp}.json'
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {output_file}")
    return results


def main():
    args = parse_args()
    
    logger.info(f"Starting experiment: {args.experiment}")
    logger.info(f"Device: {args.device}")
    
    if args.experiment == 'evaluate':
        results = run_evaluation(args)
    elif args.experiment == 'benchmark':
        results = run_benchmark(args)
    elif args.experiment == 'ablation':
        results = run_ablation(args)
    
    logger.info("Experiment completed successfully!")


if __name__ == '__main__':
    main()