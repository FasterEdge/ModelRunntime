# FasterEdge 开源项目 - Github: https://github.com/FasterEdge - Gitee: https://gitee.com/FasterEdge
# TensorFlow Lite (.tflite) 模型推理示例
# 动态参数均用注释占位，按实际模型修改

import time
import numpy as np
import tensorflow as tf

# ===== 动态参数（按实际模型修改）=====
# MODEL_PATH: 模型文件路径，例如 "yolov5s-fp16.tflite"
MODEL_PATH = "model.tflite"          # TODO: 替换为你的 .tflite 模型路径
# INPUT_SIZE: 输入张量尺寸 (H, W)
INPUT_SIZE = (640, 640)              # TODO: 按模型输入尺寸修改
# INPUT_SHAPE: 完整输入 shape，含 batch/channel
INPUT_SHAPE = (1, 640, 640, 3)       # TODO: 按模型输入 shape 修改
# NUM_CLASSES: 类别数（COCO=80）
NUM_CLASSES = 80                     # TODO: 按模型实际类别数修改
# CONF_THRESHOLD: 置信度阈值
CONF_THRESHOLD = 0.25

def load_model(path):
    # 创建解释器并分配张量
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    return interpreter

def preprocess(image_path):
    import cv2
    img = cv2.imread(image_path)
    img = cv2.resize(img, INPUT_SIZE)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)  # (1, H, W, C)
    return img

def run_inference(interpreter, data):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    # 若输入为 int8 量化，需乘/加 scale & zero_point（见 TFLite 量化参数）
    interpreter.set_tensor(input_details[0]["index"], data)  # TODO: 量化模型需反量化
    interpreter.invoke()
    # 返回所有输出
    return [interpreter.get_tensor(d["index"]) for d in output_details]

def postprocess(outputs):
    # TODO: 根据模型输出结构解析（NMS / softmax / 回归等）
    # 例如检测模型输出 (1, 25200, 5+num_classes)
    return outputs

if __name__ == "__main__":
    interpreter = load_model(MODEL_PATH)
    data = preprocess("example.png")  # TODO: 替换输入图片
    # 预热
    for _ in range(3):
        run_inference(interpreter, data)
    # 计时推理
    start = time.time()
    outputs = run_inference(interpreter, data)
    print(f"inference time: {time.time()-start:.4f}s")
    result = postprocess(outputs)
    print("done", len(result), "outputs")
