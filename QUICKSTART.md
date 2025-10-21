# YOLO Agent 快速开始指南

## 1. 安装环境

### 创建 Conda 环境
```bash
conda env create -f environment.yml
conda activate yolo_agent
```

### 或使用 pip 安装
```bash
pip install -r requirements.txt
```

## 2. 验证安装

```python
import torch
import ultralytics
import clip

print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"Ultralytics: {ultralytics.__version__}")
print("CLIP imported successfully")
```

## 3. 准备数据

确保 DOTA 数据集结构如下:
```
data/DOTA/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

## 4. 运行第一个示例

### 方法1: 使用演示脚本

```bash
# 基线检测
python scripts/demo.py --image data/DOTA/images/val/P0005.png --method baseline --show

# Agentic检测
python scripts/demo.py --image data/DOTA/images/val/P0005.png --method agentic --show
```

### 方法2: 使用 Python 代码

```python
from src.models.model_loader import ModelLoader
from src.agents.detection_agent import DetectionAgent
from src.agents.vlm_agent import VLMAgent
from src.agents.query_processor import QueryProcessor
from src.utils.visualization import Visualizer

# 加载配置和类别
import yaml
with open('configs/DOTA.yaml', 'r') as f:
    data_config = yaml.safe_load(f)
class_names = list(data_config['names'].values())

# 加载模型
model_loader = ModelLoader('configs/model_configs.yaml')
yolo_model = model_loader.load_yolo(device='cuda')
vlm_model = model_loader.load_vlm(device='cuda')

# 创建Agent
vlm_agent = VLMAgent(vlm_model, verification_threshold=0.3)
query_processor = QueryProcessor(class_names)
agent = DetectionAgent(
    yolo_model=yolo_model,
    vlm_agent=vlm_agent,
    query_processor=query_processor,
    enable_vlm=True
)

# 执行检测
image_path = 'data/DOTA/images/val/P0005.png'
results = agent.detect(image_path, query="find all planes")

# 可视化
visualizer = Visualizer(class_names)
visualizer.draw_detections(image_path, results, show=True)

print(f"检测到 {len(results['boxes'])} 个目标")
for i, name in enumerate(results['class_names']):
    print(f"  {name}: {results['scores'][i]:.3f}")
```

## 5. 运行评估

```bash
# 在验证集上评估 (限制100张图像)
python scripts/evaluate.py --method compare --max-images 100
```

## 6. 查看结果

结果将保存在 `outputs/` 目录下:
- `outputs/visualizations/` - 可视化结果
- `outputs/predictions/` - 预测结果
- `outputs/logs/` - 运行日志

## 7. 常见问题

### Q: CUDA out of memory 错误
A: 减小批处理大小或使用较小的模型(如 yolov8n)

### Q: CLIP 模型下载慢
A: 设置代理或手动下载模型文件

### Q: 找不到数据集
A: 检查数据集路径配置是否正确

## 8. 下一步

- 阅读完整文档: `README.md`
- 运行完整实验: `python run_experiments.py --experiment evaluate`
- 自定义配置: 修改 `configs/` 下的配置文件
- 训练自己的模型: `python scripts/train.py`

## 9. 获取帮助

查看脚本帮助信息:
```bash
python scripts/demo.py --help
python scripts/evaluate.py --help
python scripts/train.py --help
```

祝你使用愉快! 🎉
