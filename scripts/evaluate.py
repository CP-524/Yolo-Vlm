"""
DOTA 数据集统一评估脚本
支持:
- YOLO 基线方法评估
- Agentic 方法评估 (YOLO + VLM验证)
- 两种方法对比
- 超参数优化
"""
import argparse
import logging
import json
import time
from pathlib import Path
import sys
import itertools

sys.path.append(str(Path(__file__).parent.parent))

from src.models.model_loader import ModelLoader
from src.agents.detection_agent import DetectionAgent
from src.agents.vlm_agent import VLMAgent
from src.agents.query_processor import QueryProcessor
from src.evaluation.dota_evaluator import DOTAEvaluator
from src.utils.data_utils import load_yaml_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description='DOTA 数据集统一评估脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 评估 YOLO 基线方法
  python scripts/evaluate.py --method yolo --max-images 100
  
  # 评估 Agentic 方法
  python scripts/evaluate.py --method agent --max-images 100
  
  # 对比两种方法
  python scripts/evaluate.py --method compare --max-images 50
  
  # 超参数优化
  python scripts/evaluate.py --method optimize --max-images 50
  
  # 完整验证集评估
  python scripts/evaluate.py --method yolo --split val
        """
    )
    
    parser.add_argument('--model-config', type=str, 
                       default='configs/model_configs.yaml',
                       help='模型配置文件路径')
    parser.add_argument('--pipeline-config', type=str, 
                       default='configs/pipeline_configs.yaml',
                       help='Pipeline 配置文件路径')
    parser.add_argument('--data-config', type=str, 
                       default='configs/DOTA.yaml',
                       help='数据集配置文件路径')
    parser.add_argument('--data-root', type=str, 
                       default='data/DOTA',
                       help='数据集根目录')
    parser.add_argument('--split', type=str, 
                       default='val',
                       choices=['train', 'val', 'test'],
                       help='数据集划分')
    parser.add_argument('--method', type=str, 
                       default='yolo',
                       choices=['yolo', 'agent', 'compare', 'optimize'],
                       help='评估方法: yolo(基线), agent(Agentic), compare(对比), optimize(超参数优化)')
    parser.add_argument('--query', type=str, 
                       default=None,
                       help='自然语言查询(仅用于 agent 方法)')
    parser.add_argument('--use-auto-query', 
                       action='store_true',
                       help='自动从真实标注生成查询')
    parser.add_argument('--max-images', type=int, 
                       default=None,
                       help='评估的最大图片数量(None 表示全部)')
    parser.add_argument('--device', type=str, 
                       default='cuda',
                       choices=['cuda', 'cpu'],
                       help='计算设备')
    parser.add_argument('--conf-thresh', type=float, 
                       default=0.25,
                       help='置信度阈值')
    parser.add_argument('--iou-thresh', type=float, 
                       default=0.45,
                       help='NMS IoU 阈值')
    parser.add_argument('--eval-iou-thresh', type=float, 
                       default=0.5,
                       help='评估时的 IoU 阈值')
    parser.add_argument('--output-dir', type=str,
                       default='outputs/evaluation',
                       help='结果输出目录')
    
    # 超参数优化相关参数
    parser.add_argument('--conf-range', type=float, nargs='+',
                       default=[0.2, 0.25, 0.3],
                       help='置信度阈值搜索范围 (用于 optimize 模式)')
    parser.add_argument('--vlm-range', type=float, nargs='+',
                       default=[0.15, 0.18, 0.20, 0.25],
                       help='VLM 验证阈值搜索范围 (用于 optimize 模式)')
    
    return parser.parse_args()


def save_results(results, args, elapsed_time):
    """保存评估结果"""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f'{args.method}_evaluation_{args.split}_{timestamp}.json'
    
    num_images = args.max_images if args.max_images else len(results.get('class_metrics', {}))
    
    results_json = {
        'method': args.method,
        'split': args.split,
        'timestamp': timestamp,
        'metrics': {
            'mAP': float(results.get('map', 0)),
            'mAP@50': float(results.get('map50', 0)),
            'mAP@75': float(results.get('map75', 0)),
            'precision': float(results.get('mean_precision', 0)),
            'recall': float(results.get('mean_recall', 0)),
            'f1': float(results.get('mean_f1', 0))
        },
        'per_class_metrics': {
            cls: {
                'precision': float(metrics['precision']),
                'recall': float(metrics['recall']),
                'f1': float(metrics['f1'])
            }
            for cls, metrics in results.get('class_metrics', {}).items()
        },
        'performance': {
            'total_time_seconds': elapsed_time,
            'num_images': num_images,
            'fps': num_images / elapsed_time if elapsed_time > 0 else 0,
            'time_per_image_ms': elapsed_time * 1000 / num_images if num_images > 0 else 0
        },
        'config': {
            'conf_threshold': args.conf_thresh,
            'iou_threshold': args.iou_thresh,
            'eval_iou_threshold': args.eval_iou_thresh,
            'device': args.device
        }
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n结果已保存到: {output_file}")
    return output_file


def print_performance_stats(elapsed_time, num_images):
    """打印性能统计"""
    print("\n" + "="*80)
    print("性能统计")
    print("="*80)
    print(f"总耗时:       {elapsed_time:.2f} 秒")
    print(f"图片数量:     {num_images}")
    print(f"处理速度:     {num_images/elapsed_time:.2f} 图片/秒")
    print(f"单图耗时:     {elapsed_time*1000/num_images:.1f} 毫秒")
    print("="*80)


def optimize_hyperparameters(args):
    """超参数优化"""
    logger.info("\n" + "="*80)
    logger.info("超参数优化模式")
    logger.info("="*80)
    
    # 加载配置
    model_config = load_yaml_config(args.model_config)
    pipeline_config = load_yaml_config(args.pipeline_config)
    data_config = load_yaml_config(args.data_config)
    
    class_names = list(data_config['names'].values())
    
    # 加载模型(只加载一次)
    logger.info("加载 YOLO 模型...")
    model_loader = ModelLoader(args.model_config)
    yolo_model = model_loader.load_yolo(device=args.device)
    
    logger.info("加载 VLM 模型...")
    vlm_model = model_loader.load_vlm(device=args.device)
    
    # 定义搜索空间
    conf_thresholds = args.conf_range
    vlm_thresholds = args.vlm_range
    
    all_results = []
    total_combinations = len(conf_thresholds) * len(vlm_thresholds)
    
    logger.info(f"\n搜索空间:")
    logger.info(f"  置信度阈值: {conf_thresholds}")
    logger.info(f"  VLM 阈值:   {vlm_thresholds}")
    logger.info(f"  总组合数:   {total_combinations}")
    logger.info(f"  每组评估:   {args.max_images or 'all'} 张图片\n")
    
    # 网格搜索
    for i, (conf_th, vlm_th) in enumerate(itertools.product(conf_thresholds, vlm_thresholds)):
        logger.info(f"[{i+1}/{total_combinations}] 测试 conf={conf_th}, vlm={vlm_th}")
        
        # 更新YOLO置信度
        yolo_model.conf_threshold = conf_th
        
        # 创建VLM Agent
        vlm_agent = VLMAgent(vlm_model, verification_threshold=vlm_th)
        
        # 创建检测Agent
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
            split=args.split
        )
        
        # 评估
        start_time = time.time()
        metrics = evaluator.evaluate(
            use_agentic=True,
            max_images=args.max_images,
            iou_threshold=args.eval_iou_thresh
        )
        elapsed = time.time() - start_time
        
        # 保存结果
        result = {
            'conf_threshold': conf_th,
            'vlm_threshold': vlm_th,
            'mAP': metrics['map'],
            'mAP@50': metrics['map50'],
            'mAP@75': metrics['map75'],
            'precision': metrics['mean_precision'],
            'recall': metrics['mean_recall'],
            'f1': metrics['mean_f1'],
            'time_seconds': elapsed
        }
        all_results.append(result)
        
        logger.info(f"  结果: mAP={metrics['map']:.4f}, P={metrics['mean_precision']:.4f}, "
                   f"R={metrics['mean_recall']:.4f}, F1={metrics['mean_f1']:.4f} ({elapsed:.1f}s)")
    
    # 找到最佳配置
    best_result = max(all_results, key=lambda x: x['mAP'])
    
    logger.info("\n" + "="*80)
    logger.info("优化结果")
    logger.info("="*80)
    logger.info(f"最佳配置:")
    logger.info(f"  置信度阈值:     {best_result['conf_threshold']}")
    logger.info(f"  VLM 验证阈值:   {best_result['vlm_threshold']}")
    logger.info(f"  mAP:           {best_result['mAP']:.4f}")
    logger.info(f"  mAP@50:        {best_result['mAP@50']:.4f}")
    logger.info(f"  mAP@75:        {best_result['mAP@75']:.4f}")
    logger.info(f"  Precision:     {best_result['precision']:.4f}")
    logger.info(f"  Recall:        {best_result['recall']:.4f}")
    logger.info(f"  F1:            {best_result['f1']:.4f}")
    logger.info("="*80)
    
    # 保存结果
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f'optimization_results_{timestamp}.json'
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'best': best_result,
            'all_results': all_results,
            'search_space': {
                'conf_thresholds': conf_thresholds,
                'vlm_thresholds': vlm_thresholds
            }
        }, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n结果已保存到: {output_file}\n")
    
    return best_result


def main():
    args = parse_args()
    
    print("\n" + "="*80)
    print(f"DOTA 数据集评估 - {args.method.upper()} 方法")
    print("="*80)
    
    # 加载配置
    logger.info("加载配置文件...")
    model_config = load_yaml_config(args.model_config)
    data_config = load_yaml_config(args.data_config)
    
    # 更新配置
    if 'yolo' in model_config:
        model_config['yolo']['confidence_threshold'] = args.conf_thresh
        model_config['yolo']['iou_threshold'] = args.iou_thresh
        model_config['yolo']['device'] = args.device
    
    # 获取类别名称
    class_names = list(data_config['names'].values())
    logger.info(f"类别数量: {len(class_names)}")
    
    # 加载模型
    logger.info(f"加载 YOLO 模型: {model_config['yolo']['model_name']}")
    model_loader = ModelLoader(args.model_config)
    yolo_model = model_loader.load_yolo(device=args.device)
    
    # 创建 Agent 组件
    enable_vlm = args.method in ['agent', 'compare']
    vlm_agent = None
    
    if enable_vlm:
        try:
            logger.info("加载 VLM 模型...")
            pipeline_config = load_yaml_config(args.pipeline_config)
            vlm_model = model_loader.load_vlm(device=args.device)
            vlm_agent = VLMAgent(
                vlm_model,
                verification_threshold=pipeline_config['vlm_verification'].get('verification_threshold', 0.2)
            )
        except Exception as e:
            logger.warning(f"VLM 模型加载失败: {e}")
            logger.warning("继续使用纯 YOLO 方法...")
            enable_vlm = False
    
    # 创建查询处理器
    query_processor = QueryProcessor(class_names)
    
    # 创建检测 Agent
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
    logger.info(f"开始评估 {args.split} 集...")
    logger.info(f"置信度阈值: {args.conf_thresh}")
    logger.info(f"NMS IoU 阈值: {args.iou_thresh}")
    logger.info(f"评估 IoU 阈值: {args.eval_iou_thresh}")
    
    if args.max_images:
        logger.info(f"评估图片数量: {args.max_images}")
    else:
        logger.info("评估全部图片")
    
    start_time = time.time()
    
    try:
        # 超参数优化模式
        if args.method == 'optimize':
            optimize_hyperparameters(args)
            return
        
        # 标准评估模式
        if args.method == 'compare':
            results = evaluator.compare_methods(
                query=args.query,
                max_images=args.max_images,
                use_auto_query=args.use_auto_query
            )
        elif args.method == 'yolo':
            results = evaluator.evaluate(
                use_agentic=False,
                max_images=args.max_images,
                iou_threshold=args.eval_iou_thresh
            )
        else:  # agent
            results = evaluator.evaluate(
                use_agentic=True,
                query=args.query,
                max_images=args.max_images,
                use_auto_query=args.use_auto_query,
                iou_threshold=args.eval_iou_thresh
            )
        
        elapsed_time = time.time() - start_time
        num_images = args.max_images if args.max_images else 373  # val 集总数
        
        # 保存结果
        output_file = save_results(results, args, elapsed_time)
        
        # 打印性能统计
        print_performance_stats(elapsed_time, num_images)
        
        print("\n" + "="*80)
        print("评估完成!")
        print("="*80 + "\n")
        
    except KeyboardInterrupt:
        logger.warning("\n评估被用户中断")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n评估过程中出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
