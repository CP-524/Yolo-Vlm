# YOLO Agent - Agentic Object Detection Framework

基于 YOLO 和视觉语言模型(VLM)的 Agentic 目标检测框架，用于 DOTA 数据集的旋转目标检测。

## 📁 项目结构

```
yolo_agent/
├── 📁 configs/                    # 配置文件
│   ├── model_configs.yaml        # 模型配置（YOLO、VLM选择等）
│   ├── pipeline_configs.yaml     # Pipeline参数配置
│   └── DOTA.yaml                 # DOTA数据集配置
├── 📁 src/                       # 源代码
│   ├── __init__.py
│   ├── agents/                   # Agent核心模块
│   │   ├── __init__.py
│   │   ├── detection_agent.py    # 检测Agent（文章中的Pipeline）
│   │   ├── vlm_agent.py         # VLM代理模块
│   │   └── query_processor.py    # 查询处理模块
│   ├── models/                   # 模型加载和管理
│   │   ├── __init__.py
│   │   ├── model_loader.py       # 统一模型加载
│   │   ├── yolo_wrapper.py       # YOLO封装
│   │   └── vlm_wrapper.py        # VLM封装（CLIP、LLaVA等）
│   ├── utils/                    # 工具函数
│   │   ├── __init__.py
│   │   ├── visualization.py      # 可视化工具
│   │   ├── metrics.py            # 指标计算
│   │   ├── parallel.py           # 并行处理
│   │   └── data_utils.py         # 数据预处理
│   └── evaluation/               # 评估模块
│       ├── __init__.py
│       ├── dota_evaluator.py     # DOTA数据集评估
│       ├── dota_metrics.py       # DOTA指标
│       └── benchmark.py          # 基准测试
├── 📁 scripts/                   # 运行脚本
│   ├── train.py                  # 训练脚本
│   ├── evaluate.py               # 评估脚本
│   ├── demo.py                   # 演示脚本
│   ├── optimize.py               # 优化脚本
│   ├── interactive_demo.py       # 交互式演示脚本
│   └── quick_demo.py             # 快速演示脚本
├── 📁 data/                      # 数据目录
│   └── DOTA/                     # DOTA数据集
│       ├── images/               # 原始图像
│       ├── labels/               # 标注文件
│       └── all.yaml              # 数据集配置
├── 📁 outputs/                   # 输出结果
│   ├── checkpoints/              # 模型检查点
│   ├── predictions/              # 预测结果
│   ├── logs/                     # 训练日志
│   └── visualizations/           # 可视化结果
├── 📁 experiments/               # 实验记录
│   ├── evaluation/               # 评估实验
│   ├── benchmark/                # 基准测试
│   └── ablation/                 # 消融实验
├── requirements.txt              # 依赖包
├── environment.yml               # Conda环境
├── README.md                     # 项目说明
├── QUICKSTART.md                 # 快速开始指南
├── PROJECT_SUMMARY.md            # 项目总结
└── run_experiments.py            # 实验运行入口
```

## 🚀 快速开始

### 1. 环境安装

#### 使用 Conda (推荐)
```bash
conda env create -f environment.yml
conda activate yolo_agent
```

#### 使用 pip
```bash
pip install -r requirements.txt
```

**重要**: 确保安装CLIP模块：
```bash
pip install clip-openai
```

### 2. 数据准备

下载 DOTA 数据集并放置在 `data/DOTA/` 目录下:
```
data/DOTA/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

### 3. 模型准备

项目已包含预训练模型 `yolov8s.pt`，CLIP模型会自动下载。

## 📖 使用方法

### 演示 - 单张图像检测

```bash
# YOLO基线方法
python scripts/demo.py --image data/DOTA/images/val/P0005.png --method yolo --show

# Agentic方法
python scripts/demo.py --image data/DOTA/images/val/P0005.png --method agent --show

# 使用自然语言查询
python scripts/demo.py --image data/DOTA/images/val/P0005.png --method agent --query "find all planes" --show

# 对比两种方法
python scripts/demo.py --image data/DOTA/images/val/P0005.png --method compare --show
```

### 交互式演示（推荐）

#### 快速演示脚本
```bash
# 编辑 scripts/quick_demo.py 中的配置参数
python scripts/quick_demo.py
```

#### 完整交互式演示
```bash
python scripts/interactive_demo.py
```

### 训练

```bash
python scripts/train.py \
    --config configs/model_configs.yaml \
    --data configs/DOTA.yaml \
    --epochs 100 \
    --batch-size 16 \
    --device cuda
```

### 评估

```bash
# 评估YOLO基线方法
python scripts/evaluate.py --method yolo --max-images 100

# 评估Agentic方法
python scripts/evaluate.py --method agent --max-images 100

# 对比两种方法
python scripts/evaluate.py --method compare --max-images 100
```

### 运行完整实验

```bash
# 评估实验
python run_experiments.py --experiment evaluate --max-images 200

# 性能基准测试
python run_experiments.py --experiment benchmark --max-images 100

# 消融实验
python run_experiments.py --experiment ablation --max-images 100
```

### 超参数优化

```bash
python scripts/optimize.py --max-images 100 --output outputs/optimization_results.json
```

## 🔧 配置说明

### 模型配置 (`configs/model_configs.yaml`)

```yaml
yolo:
  model_name: "YOLO/DOTA.pt"  # 或 yolov8s.pt
  pretrained: true
  confidence_threshold: 0.25
  iou_threshold: 0.45
  max_det: 300
  device: "cuda"
  imgsz: 1024

vlm:
  model_type: "clip"
  model_name: "ViT-B/32"
  device: "cuda"
  batch_size: 32

clip:
  model_name: "ViT-B/32"
  pretrained: "openai"
```

### Pipeline配置 (`configs/pipeline_configs.yaml`)

```yaml
agentic_pipeline:
  enable_vlm: true
  enable_query_processor: true
  use_parallel: true
  max_workers: 4

query_processing:
  similarity_threshold: 0.3
  top_k_candidates: 10
  use_semantic_matching: true

vlm_verification:
  enable: true
  verification_threshold: 0.2  # 已优化，避免过度过滤
  batch_size: 32
  use_ensemble: false
```

## 📊 核心功能

### 1. 检测Agent

集成 YOLO、VLM 和查询处理的完整 pipeline:

```python
from src.models.model_loader import ModelLoader
from src.agents.detection_agent import DetectionAgent

# 加载模型
model_loader = ModelLoader('configs/model_configs.yaml')
yolo_model = model_loader.load_yolo()
vlm_model = model_loader.load_vlm()

# 创建Agent
agent = DetectionAgent(yolo_model, vlm_model, query_processor)

# 检测
results = agent.detect('image.jpg', query="find all planes")
```

### 2. VLM验证

使用 CLIP 等视觉语言模型验证和优化检测结果:

```python
from src.agents.vlm_agent import VLMAgent

vlm_agent = VLMAgent(vlm_model, verification_threshold=0.2)
verified_results = vlm_agent.verify_detections(image_path, detections)
```

### 3. 查询处理

支持自然语言查询:

```python
from src.agents.query_processor import QueryProcessor

processor = QueryProcessor(class_names)
query_info = processor.parse_query("find all planes in the image")
filtered_results = processor.filter_by_query(detections, query_info)
```

### 4. 评估和可视化

```python
from src.evaluation.dota_evaluator import DOTAEvaluator
from src.utils.visualization import Visualizer

# 评估
evaluator = DOTAEvaluator(agent, data_root, class_names)
metrics = evaluator.compare_methods()

# 可视化
visualizer = Visualizer(class_names)
visualizer.draw_detections(image_path, results, show=True)
```

## 🎯 Agentic vs YOLO 对比

### Agentic方法的优势：

1. **零样本检测**: 可以检测训练时未见过的类别
2. **语义理解**: 通过自然语言查询进行精确检测
3. **遮挡检测**: 能够检测部分遮挡的目标
4. **质量过滤**: 通过VLM验证提高检测质量

### 检测结果特点：

- **Agentic检测数量 ≤ YOLO检测数量**: Agentic方法会过滤和优化YOLO的原始结果
- **质量优先**: 注重检测精度而非数量
- **语义相关**: 结果更符合查询意图

## 🔍 支持的查询类型

### 类别查询
```bash
--query "find all planes"
--query "detect ships"
--query "locate vehicles"
```

### 属性查询
```bash
--query "find large vehicles"
--query "detect small objects"
--query "locate round objects"
```

### 场景查询
```bash
--query "find objects in water"
--query "detect aerial vehicles"
--query "locate ground structures"
```

### 复合查询
```bash
--query "find all planes and helicopters"
--query "detect vehicles and ships"
--query "locate sports facilities"
```

## 📝 DOTA 数据集类别

支持 15 个类别:
- baseball-diamond (棒球场)
- basketball-court (篮球场)
- bridge (桥梁)
- ground-track-field (田径场)
- harbor (港口)
- helicopter (直升机)
- large-vehicle (大型车辆)
- plane (飞机)
- roundabout (环岛)
- ship (船只)
- small-vehicle (小型车辆)
- soccer-ball-field (足球场)
- storage-tank (储罐)
- swimming-pool (游泳池)
- tennis-court (网球场)

## 🔬 技术特性

- ✅ YOLO 目标检测 (YOLOv8/YOLO11)
- ✅ CLIP 视觉语言模型验证 (clip-openai 1.0.post20230121)
- ✅ 自然语言查询处理
- ✅ 多种评估指标 (mAP, Precision, Recall, F1)
- ✅ 可视化工具
- ✅ 性能基准测试
- ✅ 超参数优化
- ✅ 消融实验支持
- ✅ 交互式演示脚本
- ✅ 并行处理支持

## 🛠️ 故障排除

### CLIP导入错误
```bash
# 如果遇到 ModuleNotFoundError: No module named 'clip'
pip install clip-openai
```

### VLM验证无结果
- 检查 `configs/pipeline_configs.yaml` 中的 `verification_threshold`
- 建议设置为 0.2 或更低

### 内存不足
- 减少 `batch_size` 设置
- 使用 `device: "cpu"` 而非 `cuda`

## 📈 性能优化

### 推荐配置
- **GPU**: CUDA 11.7+ 推荐
- **内存**: 8GB+ RAM
- **存储**: SSD 推荐

### 参数调优
- `verification_threshold`: 0.2-0.3 (平衡精度和召回)
- `batch_size`: 16-32 (根据GPU内存调整)
- `max_workers`: 4-8 (CPU核心数)

## 🤝 贡献

欢迎提交 Issue 和 Pull Request!

## 📄 许可证

MIT License

## 📧 联系方式

如有问题，请提交 Issue 或联系项目维护者。

---

## 📚 相关文档

- [QUICKSTART.md](QUICKSTART.md) - 快速开始指南
- [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) - 项目总结
- [scripts/interactive_demo.py](scripts/interactive_demo.py) - 交互式演示
- [scripts/quick_demo.py](scripts/quick_demo.py) - 快速演示