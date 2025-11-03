# YOLO Agent - 智能目标检测框架

基于 YOLO 和视觉语言模型(VLM)的智能目标检测系统，结合自然语言理解能力，提升检测的准确性和灵活性。

## 📌 项目简介

本项目实现了两种目标检测方法：

- **YOLO 基线方法**: 传统的 YOLO 目标检测
- **Agentic 方法**: 结合 VLM 和自然语言查询的智能检测系统

**核心创新**：通过 CLIP 等视觉语言模型对 YOLO 检测结果进行语义验证和过滤，支持自然语言查询，提高检测质量。

**当前应用**：在 DOTA 数据集上验证，支持 15 个类别的目标检测。

**扩展性**：框架设计通用化，可轻松扩展到其他检测任务和数据集。

## 🎯 主要功能

- ✅ YOLO 系列模型支持 (YOLOv8/YOLO11)
- ✅ CLIP 视觉语言模型验证
- ✅ 自然语言查询检测
- ✅ 完整的评估指标 (mAP@50, mAP@75, Precision, Recall, F1)
- ✅ 可视化检测结果
- ✅ 模块化设计，易于扩展

## 📁 项目结构

```text
yolo_agent/
├── configs/          # 配置文件
│   ├── model_configs.yaml      # 模型配置
│   ├── pipeline_configs.yaml   # Pipeline配置
│   └── DOTA.yaml               # 数据集配置
├── src/              # 源代码
│   ├── agents/       # Agent 核心模块
│   │   ├── detection_agent.py  # 检测Agent
│   │   ├── vlm_agent.py        # VLM验证模块
│   │   └── query_processor.py  # 查询处理器
│   ├── models/       # 模型封装
│   ├── evaluation/   # 评估工具
│   └── utils/        # 工具函数
├── scripts/          # 运行脚本
│   ├── evaluate.py   # 主评估脚本
│   └── quick_demo.py # 快速演示
├── data/             # 数据集目录
└── outputs/          # 输出结果
    └── evaluation/   # 评估结果
```

## 🚀 快速开始

### 1. 环境安装

```bash
# 安装依赖
pip install -r requirements.txt

# 安装 CLIP (VLM功能必需)
pip install clip-openai
```

**主要依赖:**

- Python 3.8+
- PyTorch 2.0+
- ultralytics 8.3+
- clip-openai
- shapely 2.0+

### 2. 数据准备

**以 DOTA 数据集为例：**

```text
data/DOTA/
├── images/val/  # 验证集图片 (373张)
└── labels/val/  # 验证集标注
```

**扩展到其他数据集：**

只需按照相同结构组织数据，并修改 `configs/` 中的配置文件即可。

### 3. 模型准备

将训练好的 YOLO 模型放在指定路径，并在 `configs/model_configs.yaml` 中配置模型路径。

**当前配置示例：** `YOLO/DOTA.pt` (YOLOv8 模型)

## 📖 使用方法

### 评估检测性能

**评估 YOLO 基线方法 (推荐先运行):**

```bash
# 快速测试 (50张图片)
python scripts/evaluate.py --method yolo --max-images 50

# 完整验证集评估 (373张图片)
python scripts/evaluate.py --method yolo --split val
```

**评估 Agentic 方法:**

```bash
# 使用 VLM 验证
python scripts/evaluate.py --method agent --max-images 50

# 对比两种方法
python scripts/evaluate.py --method compare --max-images 50
```

**自定义评估参数:**

```bash
python scripts/evaluate.py \
    --method yolo \
    --split val \
    --max-images 100 \
    --conf-thresh 0.25 \
    --iou-thresh 0.45 \
    --eval-iou-thresh 0.5 \
    --device cuda
```

### 单图像检测演示

```bash
# 编辑 scripts/quick_demo.py 中的配置
python scripts/quick_demo.py
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

## 📊 评估指标

### 主要指标

评估系统支持全面的目标检测指标：

| 指标 | 说明 | DOTA基准 |
|------|------|----------|
| **mAP@50** | IoU=0.5 时的平均精度 | 0.597 |
| **mAP@75** | IoU=0.75 时的平均精度 | - |
| **Precision** | 检测精度 | 0.797 |
| **Recall** | 检测召回率 | 0.527 |
| **F1 Score** | 精度和召回率的调和平均 | - |

### 评估输出示例

```text
================================================================================
DOTA EVALUATION METRICS
================================================================================
mAP@50:     0.5929
mAP@75:     0.4501
Precision:  0.6496
Recall:     0.7068
F1:         0.6627

Per-class metrics:
--------------------------------------------------------------------------------
Class                Precision    Recall       F1           Pred/GT
--------------------------------------------------------------------------------
plane                0.9748       0.9644       0.9696       278/281
ship                 0.5714       0.6829       0.6222       49/41
large-vehicle        0.8776       0.8281       0.8521       686/727
small-vehicle        0.4301       0.7288       0.5410       1481/874
...
================================================================================
```

### 结果保存

评估结果自动保存为 JSON 格式，包含详细的性能统计：

```json
{
  "metrics": {
    "mAP": 0.5215,
    "mAP@50": 0.5929,
    "mAP@75": 0.4501,
    "precision": 0.6496,
    "recall": 0.7068,
    "f1": 0.6627
  },
  "per_class_metrics": {...},
  "performance": {
    "total_time_seconds": 17.03,
    "num_images": 50,
    "fps": 2.94
  }
}
```

## 🔧 核心功能

### 1. 检测 Agent

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

### Agentic 方法的优势

1. **语义理解**: 通过自然语言查询进行精确检测
2. **质量过滤**: 使用 VLM 验证提高检测质量
3. **灵活性**: 支持复杂查询和多类别组合
4. **可解释性**: 结果更符合用户意图

### 检测结果特点

- **质量优先**: Agentic 方法会过滤 YOLO 原始结果，提高精度
- **语义相关**: 结果更符合查询意图
- **适应性强**: 可根据不同场景调整

## 🔍 自然语言查询示例

### 类别查询

```bash
python scripts/quick_demo.py --method agent --query "find all planes"
python scripts/quick_demo.py --method agent --query "detect ships"
```

### 复合查询

```bash
python scripts/quick_demo.py --method agent --query "find all planes and helicopters"
python scripts/quick_demo.py --method agent --query "detect vehicles and ships"
```

### 属性查询

```bash
python scripts/quick_demo.py --method agent --query "find large vehicles"
python scripts/quick_demo.py --method agent --query "locate round objects"
```

## 📝 当前支持的类别 (DOTA数据集)

15 个目标类别：plane, ship, storage-tank, baseball-diamond, tennis-court, basketball-court, ground-track-field, harbor, bridge, large-vehicle, small-vehicle, helicopter, roundabout, soccer-ball-field, swimming-pool

**扩展到其他数据集：**

框架采用模块化设计，只需修改配置文件即可适配新数据集：

1. 准备数据集 (images + labels)
2. 创建数据集配置文件 (参考 `configs/DOTA.yaml`)
3. 训练或加载 YOLO 模型
4. 运行评估脚本

## 🛠️ 常见问题

**Q: 如何扩展到其他检测任务？**

A: 框架设计通用化，支持标准 YOLO 格式数据集。只需准备数据和配置文件即可。

**Q: CLIP 导入错误？**

```bash
pip install clip-openai
```

**Q: 评估速度慢？**

- 使用 `--max-images` 限制图片数量
- 确保使用 GPU: `--device cuda`
- 减小输入图像尺寸

**Q: 内存不足？**

- 使用 CPU: `--device cpu`
- 减少 `batch_size`

## 📈 性能参考

**推荐配置:**

- GPU: CUDA 11.7+
- 内存: 8GB+ RAM
- 存储: SSD

**性能数据 (DOTA 验证集):**

- 处理速度: ~3-6 FPS (取决于GPU)
- 50张图片评估: ~20秒
- 完整验证集(373张): ~3-5分钟

**参数调优建议:**

- `verification_threshold`: 0.2-0.3 (VLM 验证阈值)
- `conf_threshold`: 0.25-0.3 (检测置信度)
- `batch_size`: 16-32 (根据内存调整)

## 📄 许可证

MIT License

## 📧 联系与贡献

欢迎提交 Issue 和 Pull Request！

---

**快速开始命令:**

```bash
# 安装依赖
pip install -r requirements.txt && pip install clip-openai

# 快速评估 (50张图片)
python scripts/evaluate.py --method yolo --max-images 50

# 完整评估 (373张图片)
python scripts/evaluate.py --method yolo --split val
```
