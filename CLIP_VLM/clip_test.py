import clip
import torch
from PIL import Image

# 1. 加载模型 - 就这么简单！
model, preprocess = clip.load("ViT-B/32", device="cuda")

# 2. 准备图像和文本
image = preprocess(Image.open(r"pic\1.jpg")).unsqueeze(0).to("cuda")
text = clip.tokenize(["a car", "a truck", "a bicycle", "a photo of nothing relevant"]).to("cuda")

# 3. 计算相似度
with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print("相似度:", similarity.cpu().numpy())