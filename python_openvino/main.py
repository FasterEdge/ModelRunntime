# FasterEdge 开源项目 - Github: https://github.com/FasterEdge - Gitee: https://gitee.com/FasterEdge
# OpenVINO IR (.xml / .bin) 模型推理示例
# 动态参数均用注释占位，按实际模型修改

import time
import numpy as np
from openvino import Core

# ===== 动态参数（按实际模型修改）=====
# MODEL_PATH: OpenVINO IR 模型 .xml 文件路径（.bin 自动关联）
MODEL_PATH = "model.xml"             # TODO: 替换为 .xml 模型路径
# DEVICE: 推理设备（"CPU" / "GPU" / "AUTO"）
DEVICE = "CPU"                       # TODO: 按可用设备修改
# INPUT_SIZE: 输入图像尺寸 (H, W)
INPUT_SIZE = (640, 640)              # TODO: 按模型输入尺寸修改
# NUM_CLASSES: 类别数
NUM_CLASSES = 80                     # TODO: 按模型实际类别数修改

def load_model(model_path, device):
    core = Core()
    model = core.read_model(model_path)
    compiled_model = core.compile_model(model, device)
    return compiled_model

def preprocess(image_path):
    import cv2
    img = cv2.imread(image_path)
    img = cv2.resize(img, INPUT_SIZE)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # CHW
    return img[None, :, :, :]

def run_inference(compiled_model, data):
    input_key = compiled_model.input(0)  # TODO: 输入名按模型修改
    output_key = compiled_model.output(0)  # TODO: 输出名按模型修改
    return compiled_model({input_key: data})[output_key]

if __name__ == "__main__":
    compiled = load_model(MODEL_PATH, DEVICE)
    data = preprocess("example.png")  # TODO: 替换输入图片
    # 预热
    for _ in range(3):
        run_inference(compiled, data)
    start = time.time()
    outputs = run_inference(compiled, data)
    print(f"inference time: {time.time()-start:.4f}s")
    print("output shape:", outputs.shape)
