# TensorFlow 2 SavedModel 推理示例
# 动态参数均用注释占位，按实际模型修改

import time
import numpy as np
import tensorflow as tf

# ===== 动态参数（按实际模型修改）=====
# MODEL_DIR: SavedModel 目录（含 saved_model.pb + variables/）
MODEL_DIR = "./saved_model"          # TODO: 替换为 SavedModel 目录
# INPUT_SIZE: 输入图像尺寸 (H, W)
INPUT_SIZE = (640, 640)              # TODO: 按模型输入尺寸修改
# SIGNATURE_KEY: 服务签名（"serving_default" 通常不变）
SIGNATURE_KEY = "serving_default"    # TODO: 若模型签名不同则修改

def load_model(model_dir):
    return tf.saved_model.load(model_dir)

def preprocess(image_path):
    import cv2
    img = cv2.imread(image_path)
    img = cv2.resize(img, INPUT_SIZE)
    img = img.astype(np.float32) / 255.0
    return img

def run_inference(model, data):
    infer = model.signatures[SIGNATURE_KEY]
    # 输入张量名通常为 "input_1"，输出名需用 model.structured_outputs 查询
    out = infer(tf.constant(data[None, ...]))  # TODO: 输入名/shape 按模型修改
    return out

if __name__ == "__main__":
    model = load_model(MODEL_DIR)
    data = preprocess("example.png")  # TODO: 替换输入图片
    # 预热
    for _ in range(3):
        run_inference(model, data)
    start = time.time()
    outputs = run_inference(model, data)
    print(f"inference time: {time.time()-start:.4f}s")
    print(outputs)  # TODO: 解析输出
