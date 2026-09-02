# FasterEdge 开源项目 - Github: https://github.com/FasterEdge - Gitee: https://gitee.com/FasterEdge
# TorchScript (.pt) 模型推理示例
# 动态参数均用注释占位，按实际模型修改

import time
import torch
import numpy as np

# ===== 动态参数（按实际模型修改）=====
# MODEL_PATH: TorchScript 模型文件路径
MODEL_PATH = "model_scripted.pt"     # TODO: 替换为你的 .pt 模型路径
# INPUT_SIZE: 输入图像尺寸 (H, W)
INPUT_SIZE = (640, 640)              # TODO: 按模型输入尺寸修改
# DEVICE: 推理设备（"cpu" / "cuda:0"）
DEVICE = "cpu"                       # TODO: 按可用设备修改
# NUM_CLASSES: 类别数
NUM_CLASSES = 80                     # TODO: 按模型实际类别数修改

def load_model(path):
    model = torch.jit.load(path, map_location=DEVICE)
    model.eval()
    return model

def preprocess(image_path):
    import cv2
    img = cv2.imread(image_path)
    img = cv2.resize(img, INPUT_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    # CHW 格式，[N, C, H, W]
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).to(DEVICE)
    return img

def run_inference(model, data):
    with torch.no_grad():
        return model(data)

if __name__ == "__main__":
    model = load_model(MODEL_PATH)
    data = preprocess("example.png")  # TODO: 替换输入图片
    # 预热
    for _ in range(3):
        run_inference(model, data)
    if DEVICE == "cuda:0":
        torch.cuda.synchronize()
    start = time.time()
    outputs = run_inference(model, data)
    if DEVICE == "cuda:0":
        torch.cuda.synchronize()
    print(f"inference time: {time.time()-start:.4f}s")
    print("output shape:", outputs.shape)  # TODO: 解析输出
