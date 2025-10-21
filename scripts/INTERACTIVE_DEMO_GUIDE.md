# 交互式检测脚本使用说明

## 📋 脚本介绍

我为你创建了两个新的脚本，让你可以直接在程序中修改参数配置，而不需要在命令行中输入：

### 1. `scripts/interactive_demo.py` - 完整功能版本
- 面向对象设计，功能完整
- 支持单张图像检测和批量检测
- 可以动态配置所有参数

### 2. `scripts/quick_demo.py` - 快速配置版本
- 简单直接，易于使用
- 直接在代码顶部修改参数
- 适合快速测试和实验

## 🚀 使用方法

### 快速开始（推荐）

1. **打开 `scripts/quick_demo.py`**
2. **修改顶部的配置参数**：

```python
# ==================== 在这里修改你的配置 ====================

# 基本配置
IMAGE_PATH = "data/DOTA/images/val/P0005.png"  # 要检测的图像路径
METHOD = "agent"  # 检测方法: 'yolo', 'agent', 'compare'
QUERY = "find all planes"  # 查询文本（可选，None表示无查询）
DEVICE = "cuda"  # 设备: 'cuda' 或 'cpu'

# 显示配置
SHOW_RESULT = True  # 是否显示检测结果
SAVE_RESULT = True  # 是否保存检测结果图像
```

3. **运行脚本**：
```bash
python scripts/quick_demo.py
```

### 高级使用（完整功能版本）

1. **打开 `scripts/interactive_demo.py`**
2. **修改 `main()` 函数中的配置**：

```python
def main():
    # ==================== 配置参数 ====================
    
    # 基本配置
    IMAGE_PATH = "data/DOTA/images/val/P0005.png"
    METHOD = "agent"  # 'yolo', 'agent', 'compare'
    QUERY = "find all planes"
    DEVICE = "cuda"
    YOLO_WEIGHTS = None
    
    # 显示和保存配置
    SHOW_RESULT = True
    SAVE_OUTPUT = True
    
    # ==================== 执行检测 ====================
    # ... 其余代码
```

3. **运行脚本**：
```bash
python scripts/interactive_demo.py
```

## 📝 参数说明

### 基本参数

| 参数 | 类型 | 说明 | 可选值 |
|------|------|------|--------|
| `IMAGE_PATH` | str | 要检测的图像路径 | 任何有效的图像文件路径 |
| `METHOD` | str | 检测方法 | `'yolo'`, `'agent'`, `'compare'` |
| `QUERY` | str | 自然语言查询 | 任何查询文本，如 `"find all planes"` |
| `DEVICE` | str | 计算设备 | `'cuda'` 或 `'cpu'` |

### 显示参数

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `SHOW_RESULT` | bool | 是否显示检测结果 | `True` |
| `SAVE_RESULT` | bool | 是否保存结果图像 | `True` |

## 🎯 使用示例

### 示例1：YOLO基线检测
```python
IMAGE_PATH = "data/DOTA/images/val/P0005.png"
METHOD = "yolo"
QUERY = None
DEVICE = "cuda"
```

### 示例2：Agentic检测（带查询）
```python
IMAGE_PATH = "data/DOTA/images/val/P0005.png"
METHOD = "agent"
QUERY = "find all planes"
DEVICE = "cuda"
```

### 示例3：对比检测
```python
IMAGE_PATH = "data/DOTA/images/val/P0005.png"
METHOD = "compare"
QUERY = "find all vehicles"
DEVICE = "cuda"
```

### 示例4：检测特定类别
```python
IMAGE_PATH = "data/DOTA/images/val/P0003.png"
METHOD = "agent"
QUERY = "find all ships"
DEVICE = "cuda"
```

## 🔧 高级功能

### 批量检测（仅限完整版本）

在 `interactive_demo.py` 中，你可以使用 `demo_batch_detection()` 函数进行批量检测：

```python
def demo_batch_detection():
    # 图像列表
    IMAGE_PATHS = [
        "data/DOTA/images/val/P0003.png",
        "data/DOTA/images/val/P0005.png",
        "data/DOTA/images/val/P0007.png"
    ]
    
    # 对应的查询
    QUERIES = [
        "find all planes",
        "find all ships", 
        "find all vehicles"
    ]
    
    # 执行批量检测
    detector = InteractiveDetector()
    detector.setup_models(device="cuda")
    results = detector.batch_detect(
        image_paths=IMAGE_PATHS,
        method="agent",
        queries=QUERIES
    )
```

## 📊 输出结果

### 检测结果
- 检测到的目标数量
- 每个目标的类别和置信度
- 可视化图像（可选）

### 保存文件
- 结果图像保存在 `outputs/visualizations/` 目录
- 文件名格式：`{method}_{image_name}.jpg`

## ⚠️ 注意事项

1. **图像路径**：确保图像文件存在
2. **设备选择**：如果有GPU，建议使用 `'cuda'`
3. **查询文本**：使用英文查询效果更好
4. **VLM依赖**：如果CLIP模块未安装，会自动禁用VLM验证

## 🎉 优势

- ✅ **无需命令行**：直接在代码中修改参数
- ✅ **快速测试**：修改参数后直接运行
- ✅ **灵活配置**：支持所有检测方法
- ✅ **错误处理**：自动处理常见错误
- ✅ **结果可视化**：自动显示和保存结果

现在你可以方便地在程序中修改参数，而不需要记住复杂的命令行参数了！
