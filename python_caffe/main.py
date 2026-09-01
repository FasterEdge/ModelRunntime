# ─────────────────────────────────────────────────────────────
# FasterEdge 开源项目
# Github: https://github.com/FasterEdge
# Gitee:  https://gitee.com/FasterEdge
# ─────────────────────────────────────────────────────────────
# Caffe (.prototxt + .caffemodel) 模型推理示例
# 动态参数均用注释占位，按实际模型修改

import time
import numpy as np

# ===== 动态参数（按实际模型修改）=====
# PROTO_PATH: 网络结构 .prototxt 路径
PROTO_PATH = "deploy.prototxt"       # TODO: 替换为 .prototxt 路径
# WEIGHT_PATH: 权重 .caffemodel 路径
WEIGHT_PATH = "model.caffemodel"     # TODO: 替换为 .caffemodel 路径
# INPUT_SIZE: 输入图像尺寸 (H, W)
INPUT_SIZE = (224, 224)              # TODO: 按模型输入尺寸修改
# MEAN: 均值（Caffe 常用 [104,117,123] 或 [123.68,116.78,103.94]）
MEAN = [104, 117, 123]               # TODO: 按模型训练均值修改
# NUM_CLASSES: 类别数
NUM_CLASSES = 1000                   # TODO: 按模型实际类别数修改

def load_net(proto_path, weight_path):
    # Caffe 官方 pycaffe 或 cv2.dnn.readNetFromCaffe 两种方式
    import cv2
    net = cv2.dnn.readNetFromCaffe(proto_path, weight_path)
    return net

def preprocess(image_path):
    import cv2
    img = cv2.imread(image_path)
    blob = cv2.dnn.blobFromImage(img, 1.0, INPUT_SIZE, MEAN, swapRB=False, crop=False)
    return blob

def run_inference(net, blob):
    net.setInput(blob)
    # 输出层名按模型修改，常见 "prob" / "fc8"
    return net.forward()  # TODO: 指定输出层名

if __name__ == "__main__":
    net = load_net(PROTO_PATH, WEIGHT_PATH)
    blob = preprocess("example.png")  # TODO: 替换输入图片
    # 预热
    for _ in range(3):
        run_inference(net, blob)
    start = time.time()
    outputs = run_inference(net, blob)
    print(f"inference time: {time.time()-start:.4f}s")
    print("output shape:", outputs.shape)
