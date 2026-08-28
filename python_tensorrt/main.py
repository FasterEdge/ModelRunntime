# TensorRT (.engine) 模型推理示例
# 动态参数均用注释占位，按实际模型修改

import time
import numpy as np

# ===== 动态参数（按实际模型修改）=====
# ENGINE_PATH: TensorRT engine 文件路径
ENGINE_PATH = "model.engine"         # TODO: 替换为 .engine 引擎路径
# INPUT_SIZE: 输入张量尺寸 (H, W)
INPUT_SIZE = (640, 640)              # TODO: 按模型输入尺寸修改
# INPUT_DTYPE: 输入数据类型（np.float32 / np.int8 等）
INPUT_DTYPE = np.float32             # TODO: 按引擎输入类型修改
# BATCH_SIZE: 批大小
BATCH_SIZE = 1                       # TODO: 按需要修改

# TRT 的 Python 绑定在不同版本差异较大，以下使用 pycuda + tensorrt 常见写法
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False
    print("tensorrt/pycuda 未安装，仅演示代码结构")

def load_engine(engine_path):
    # TODO: 若传入 ONNX，需先经 trt.Builder 构建 engine（此处假设已构建好）
    logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def run_inference(engine, data):
    # TODO: 分配 host/device 内存，绑定 bindings，enqueue 推理
    # 输入名/输出名按 engine 的 bindings 获取（engine.get_tensor_name(i)）
    return data

if __name__ == "__main__":
    if not TRT_AVAILABLE:
        exit(0)
    engine = load_engine(ENGINE_PATH)
    data = np.random.rand(BATCH_SIZE, 3, *INPUT_SIZE).astype(INPUT_DTYPE)
    # 预热
    for _ in range(3):
        run_inference(engine, data)
    start = time.time()
    outputs = run_inference(engine, data)
    print(f"inference time: {time.time()-start:.4f}s")
