# FasterEdge 开源项目 - Github: https://github.com/FasterEdge - Gitee: https://gitee.com/FasterEdge
# PaddlePaddle (.pdmodel) 模型推理示例
# 动态参数均用注释占位，按实际模型修改

import time
import numpy as np
import paddle.inference as paddle_infer

# ===== 动态参数（按实际模型修改）=====
# MODEL_DIR: 推理模型目录（含 .pdmodel + .pdiparams）
MODEL_DIR = "./model"                # TODO: 替换为模型目录（不含文件后缀）
# MODEL_FILE: 模型文件（.pdmodel）
MODEL_FILE = "model.pdmodel"         # TODO: 按实际文件名修改
# PARAMS_FILE: 参数文件（.pdiparams）
PARAMS_FILE = "model.pdiparams"      # TODO: 按实际文件名修改
# INPUT_SIZE: 输入图像尺寸 (H, W)
INPUT_SIZE = (640, 640)              # TODO: 按模型输入尺寸修改
# NUM_CLASSES: 类别数
NUM_CLASSES = 80                     # TODO: 按模型实际类别数修改

def load_model(model_dir, model_file, params_file):
    config = paddle_infer.Config(model_dir + "/" + model_file, model_dir + "/" + params_file)
    config.disable_gpu()  # TODO: 启用 GPU 用 config.enable_use_gpu(0)
    predictor = paddle_infer.create_predictor(config)
    return predictor

def preprocess(image_path):
    import cv2
    img = cv2.imread(image_path)
    img = cv2.resize(img, INPUT_SIZE)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # CHW
    return img[None, :, :, :]

def run_inference(predictor, data):
    input_names = predictor.get_input_names()
    input_tensor = predictor.get_input_handle(input_names[0])  # TODO: 输入名按模型修改
    input_tensor.copy_from_cpu(data)
    predictor.run()
    output_names = predictor.get_output_names()
    # TODO: 输出名按模型修改，可能需要多个输出
    output_tensor = predictor.get_output_handle(output_names[0])
    return output_tensor.copy_to_cpu()

if __name__ == "__main__":
    predictor = load_model(MODEL_DIR, MODEL_FILE, PARAMS_FILE)
    data = preprocess("example.png")  # TODO: 替换输入图片
    # 预热
    for _ in range(3):
        run_inference(predictor, data)
    start = time.time()
    output = run_inference(predictor, data)
    print(f"inference time: {time.time()-start:.4f}s")
    print("output shape:", output.shape)
