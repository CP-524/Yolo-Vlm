# 评估脚本使用指南

## 📋 概述

`evaluate.py` 是统一的评估脚本，集成了所有评估功能：
- ✅ YOLO 基线方法评估
- ✅ Agentic 方法评估 (YOLO + VLM验证)
- ✅ 两种方法对比
- ✅ 超参数优化

## 🚀 快速开始

### 1. 评估 YOLO 基线方法

```bash
# 评估100张图片
python scripts/evaluate.py --method yolo --max-images 100

# 评估全部验证集
python scripts/evaluate.py --method yolo --split val

# 使用CPU
python scripts/evaluate.py --method yolo --max-images 50 --device cpu
```

### 2. 评估 Agentic 方法

```bash
# 基础评估
python scripts/evaluate.py --method agent --max-images 100

# 使用自动生成查询
python scripts/evaluate.py --method agent --max-images 100 --use-auto-query

# 使用自定义查询
python scripts/evaluate.py --method agent --max-images 100 --query "find all planes and ships"
```

### 3. 对比两种方法

```bash
# 对比基线和Agentic方法
python scripts/evaluate.py --method compare --max-images 50

# 带自动查询的对比
python scripts/evaluate.py --method compare --max-images 50 --use-auto-query
```

### 4. 超参数优化

```bash
# 使用默认搜索空间
python scripts/evaluate.py --method optimize --max-images 50

# 自定义搜索空间
python scripts/evaluate.py --method optimize --max-images 50 \
  --conf-range 0.2 0.25 0.3 \
  --vlm-range 0.15 0.18 0.20 0.25
```

## 📊 参数说明

### 基础参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--method` | 评估方法: yolo/agent/compare/optimize | yolo |
| `--max-images` | 最大评估图片数 | None (全部) |
| `--device` | 计算设备: cuda/cpu | cuda |
| `--split` | 数据集划分: train/val/test | val |

### 阈值参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--conf-thresh` | YOLO置信度阈值 | 0.25 |
| `--iou-thresh` | NMS IoU阈值 | 0.45 |
| `--eval-iou-thresh` | 评估IoU阈值 | 0.5 |

### 优化参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--conf-range` | 置信度阈值搜索范围 | [0.2, 0.25, 0.3] |
| `--vlm-range` | VLM验证阈值搜索范围 | [0.15, 0.18, 0.20, 0.25] |

### 查询参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--query` | 自然语言查询 | None |
| `--use-auto-query` | 从真实标注自动生成查询 | False |

## 📁 输出结果

所有评估结果保存在 `outputs/evaluation/` 目录：

```
outputs/evaluation/
├── yolo_evaluation_val_20251103_170504.json       # YOLO评估结果
├── agent_evaluation_val_20251103_170530.json      # Agentic评估结果
├── compare_evaluation_val_20251103_170600.json    # 对比结果
└── optimization_results_20251103_170800.json      # 优化结果
```

### 结果格式

```json
{
  "method": "yolo",
  "metrics": {
    "mAP": 0.4282,
    "mAP@50": 0.4925,
    "mAP@75": 0.3640,
    "precision": 0.5849,
    "recall": 0.6553,
    "f1": 0.6001
  },
  "per_class_metrics": {
    "plane": {
      "precision": 1.0,
      "recall": 0.9873,
      "f1": 0.9936
    }
  },
  "performance": {
    "total_time_seconds": 22.18,
    "num_images": 10,
    "fps": 0.45,
    "time_per_image_ms": 2218.4
  }
}
```

## 🎯 常用场景

### 快速测试 (5张图片)
```bash
python scripts/evaluate.py --method compare --max-images 5
```

### 中等规模验证 (50张图片)
```bash
python scripts/evaluate.py --method compare --max-images 50
```

### 完整验证集评估 (373张图片)
```bash
python scripts/evaluate.py --method compare --split val
```

### 寻找最佳超参数
```bash
# 小规模快速搜索
python scripts/evaluate.py --method optimize --max-images 30

# 中等规模精确搜索
python scripts/evaluate.py --method optimize --max-images 100
```

## ⚠️ 注意事项

1. **GPU内存**: Agent方法需要同时加载YOLO和VLM模型，确保GPU内存充足
2. **评估时间**: 
   - YOLO方法: ~0.5秒/图片
   - Agent方法: ~2秒/图片 (包含VLM验证)
   - 优化模式: 根据搜索空间大小，可能需要较长时间
3. **VLM验证**: Agent方法中VLM用于验证检测质量，会过滤掉一些低质量检测

## 📈 性能优化

评估代码已经过优化，包括：
- ✅ Bbox IoU预过滤（避免不必要的Shapely计算）
- ✅ 减少IoU阈值数量（从10个减少到2个）
- ✅ 批量处理和并行计算
- ✅ 内存优化和早停策略

100张图片评估约需1-2分钟（YOLO）或3-5分钟（Agent）。

## 🔧 故障排除

### CUDA内存不足
```bash
# 使用CPU
python scripts/evaluate.py --method yolo --device cpu

# 减少评估图片数
python scripts/evaluate.py --method compare --max-images 20
```

### 评估速度慢
```bash
# 仅评估YOLO基线（最快）
python scripts/evaluate.py --method yolo --max-images 50

# 减少优化搜索空间
python scripts/evaluate.py --method optimize --max-images 30 \
  --conf-range 0.25 0.3 \
  --vlm-range 0.18 0.20
```

### VLM过滤过多检测
```bash
# 调整VLM阈值（降低更宽松，提高更严格）
python scripts/evaluate.py --method agent --max-images 50

# 查看配置文件: configs/pipeline_configs.yaml
# vlm_verification:
#   verification_threshold: 0.18  # 调整此值
```
