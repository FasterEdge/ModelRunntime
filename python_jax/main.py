# JAX / Flax 模型推理示例
# 动态参数均用注释占位，按实际模型修改

import time
import numpy as np
import jax
import jax.numpy as jnp

# ===== 动态参数（按实际模型修改）=====
# MODEL_PATH: 模型参数/checkpoint 路径（Flax params 的 .msgpack / .npz 等）
MODEL_PATH = "params.msgpack"        # TODO: 替换为参数文件路径
# INPUT_SIZE: 输入图像尺寸 (H, W)
INPUT_SIZE = (224, 224)              # TODO: 按模型输入尺寸修改
# NUM_CLASSES: 类别数
NUM_CLASSES = 1000                   # TODO: 按模型实际类别数修改

def load_params(path):
    # Flax 推荐 .msgpack 格式；也支持 .npz / pickle（注意 pickle 安全性）
    import orbax.checkpoint as ocp
    import flax
    # TODO: 这里需按你的 checkpoint 格式与模型结构恢复
    return None

def preprocess(image_path):
    import cv2
    img = cv2.imread(image_path)
    img = cv2.resize(img, INPUT_SIZE)
    img = img.astype(np.float32) / 255.0
    return img[None, :, :, :]

@jax.jit
def predict(params, x):
    # TODO: 调用你的模型 forward（apply_fn(params, x)）
    # 此处用简单示例占位
    return x

if __name__ == "__main__":
    params = load_params(MODEL_PATH)
    data = preprocess("example.png")  # TODO: 替换输入图片
    x = jnp.asarray(data)
    # 预热
    predict(params, x).block_until_ready()
    start = time.time()
    out = predict(params, x).block_until_ready()
    print(f"inference time: {time.time()-start:.4f}s")
    print("output shape:", out.shape)
