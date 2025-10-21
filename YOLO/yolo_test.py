# 验证脚本: verify_baseline.py
from ultralytics import YOLO
import yaml

# 1. 测试YOLO加载
model = YOLO('YOLO\DOTA.pt')
print("✅ YOLO模型加载成功")

# 2. 在DOTA小样本上测试，并保存到指定目录
results = model(
    r'data\DOTA\images\val',
    project='outputs',          # 指定主保存目录为 outputs
    name='yolo_predictions',    # 在 outputs 下创建一个名为 yolo_predictions 的子目录
    save=True,                  # 保存带标注的图片
    save_txt=False               # 同时保存 .txt 格式的标注文件
)
print(f"✅ DOTA图像推理成功，结果已保存至: {results[0].save_dir}")

# 3. 验证评估流程
# ... 你的现有评估代码