# FasterEdge 开源项目 - Github: https://github.com/FasterEdge - Gitee: https://gitee.com/FasterEdge
# Keras (.h5 / .keras) 模型推理示例
# 动态参数均用注释占位，按实际模型修改

import time
import numpy as np
from tensorflow import keras

# ===== 动态参数（按实际模型修改）=====
# MODEL_PATH: 模型文件路径，例如 "model.h5"
MODEL_PATH = "model.h5"              # TODO: 替换为你的 .h5/.keras 模型路径
# INPUT_SIZE: 输入图像尺寸 (H, W)
INPUT_SIZE = (224, 224)              # TODO: 按模型输入尺寸修改
# INPUT_SHAPE: 完整输入 shape，含 batch/channel
INPUT_SHAPE = (1, 224, 224, 3)       # TODO: 按模型输入 shape 修改
# NUM_CLASSES: 分类数（ImageNet=1000）
NUM_CLASSES = 1000                   # TODO: 按模型实际类别数修改

def load_model(path):
    return keras.models.load_model(path)

def preprocess(image_path):
    import cv2
    img = cv2.imread(image_path)
    img = cv2.resize(img, INPUT_SIZE)
    img = img.astype(np.float32) / 255.0  # TODO: 归一化方式按模型训练时修改
    return np.expand_dims(img, axis=0)

def run_inference(model, data):
    return model.predict(data, verbose=0)

def postprocess(outputs):
    # TODO: 根据输出结构解析（softmax 类别 / 回归等）
    return np.argmax(outputs[0], axis=-1)  # 分类示例

if __name__ == "__main__":
    model = load_model(MODEL_PATH)
    data = preprocess("example.png")  # TODO: 替换输入图片
    # 预热
    for _ in range(3):
        run_inference(model, data)
    start = time.time()
    outputs = run_inference(model, data)
    print(f"inference time: {time.time()-start:.4f}s")
    print("pred:", postprocess(outputs))
