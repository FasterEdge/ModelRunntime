# GGUF 大模型（llama.cpp / llama-cpp-python）推理示例
# 动态参数均用注释占位，按实际模型修改

import time

# ===== 动态参数（按实际模型修改）=====
# MODEL_PATH: GGUF 模型文件路径，例如 "qwen2.5-1.5b-instruct-q4_k_m.gguf"
MODEL_PATH = "model.gguf"            # TODO: 替换为 .gguf 模型路径
# N_THREADS: 推理线程数
N_THREADS = 8                        # TODO: 按 CPU 核数修改
# MAX_TOKENS: 最大生成 token 数
MAX_TOKENS = 512                     # TODO: 按需修改
# SYSTEM_PROMPT: 系统提示词（可选）
SYSTEM_PROMPT = "You are a helpful assistant."
# TEMPERATURE: 采样温度
TEMPERATURE = 0.7                    # TODO: 按需修改

def load_model(model_path, n_threads):
    from llama_cpp import Llama
    # 这里演示 llama-cpp-python 用法；也可用 subprocess 调 llama-cli / llama-server
    return Llama(model_path=model_path, n_threads=n_threads)  # TODO: 按实际参数调整

def generate(model, prompt):
    output = model.create_chat_completion(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
    )
    return output["choices"][0]["message"]["content"]

if __name__ == "__main__":
    model = load_model(MODEL_PATH, N_THREADS)
    prompt = "Hello, how are you?"  # TODO: 替换输入提示词
    # 预热
    generate(model, "ping")
    start = time.time()
    result = generate(model, prompt)
    print(f"generate time: {time.time()-start:.4f}s")
    print(result)
