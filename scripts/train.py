"""训练脚本"""
import argparse
import logging
from pathlib import Path
import yaml

from src.models.model_loader import ModelLoader
from src.utils.data_utils import load_yaml_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Train YOLO model on DOTA dataset')
    parser.add_argument('--config', type=str, default='configs/model_configs.yaml',
                       help='Path to model config file')
    parser.add_argument('--data', type=str, default='configs/DOTA.yaml',
                       help='Path to dataset config file')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--imgsz', type=int, default=1024,
                       help='Image size')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')
    parser.add_argument('--project', type=str, default='outputs',
                       help='Project directory')
    parser.add_argument('--name', type=str, default='yolo_dota',
                       help='Experiment name')
    parser.add_argument('--resume', action='store_true',
                       help='Resume training from last checkpoint')
    return parser.parse_args()


def main():
    args = parse_args()
    
    logger.info("Starting YOLO training")
    logger.info(f"Config: {args.config}")
    logger.info(f"Data: {args.data}")
    
    # 加载模型
    model_loader = ModelLoader(args.config)
    yolo_model = model_loader.load_yolo(device=args.device)
    
    # 训练参数
    train_kwargs = {
        'epochs': args.epochs,
        'batch': args.batch_size,
        'imgsz': args.imgsz,
        'project': args.project,
        'name': args.name,
        'resume': args.resume,
        'save': True,
        'plots': True
    }
    
    # 开始训练
    logger.info("Starting training...")
    results = yolo_model.train(
        data_yaml=args.data,
        **train_kwargs
    )
    
    logger.info("Training completed!")
    logger.info(f"Results saved to {args.project}/{args.name}")


if __name__ == '__main__':
    main()
