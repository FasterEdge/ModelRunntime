# ModelRunntime

多种语言、多种格式的模型推理运行时示例集。

> 说明：每个子目录是一个独立的模型运行时示例。**所有与模型强相关的动态参数（模型路径、输入/输出张量名、输入尺寸、类别数、量化参数、设备等）均以 `TODO` 注释占位**，请按实际模型填写后使用。示例中的 `example.png` / `example2.png` 等输入图片与模型文件需自行准备。

## 目录索引

### Python

| 目录 | 模型格式 | 运行库/框架 | 说明 |
|------|----------|-------------|------|
| [python_onnx](python_onnx/) | ONNX (`.onnx`) | onnxruntime | YOLOv5n 目标检测完整示例（含 NMS 与可视化） |
| [python_pb](python_pb/) | TensorFlow (`.pb`) | tensorflow>=2.16 | YOLOv3-ResNet18 目标检测（tf.compat.v1 冻结图），附 Dockerfile |
| [python_pth_pt](python_pth_pt/) | PyTorch (`.pt` / `.pth`) | torch | YOLOv5s 权重加载 + 前向 + NMS 后处理（checkpoint 含模型结构，需 `models` 包，见目录内 requirements） |
| [python_tflite](python_tflite/) | TensorFlow Lite (`.tflite`) | tensorflow | TFLite 解释器推理（含量化提示） |
| [python_tf_savedmodel](python_tf_savedmodel/) | TF2 SavedModel | tensorflow | SavedModel 签名推理 |
| [python_keras_h5](python_keras_h5/) | Keras (`.h5` / `.keras`) | keras | Keras 模型加载与预测 |
| [python_torchscript](python_torchscript/) | TorchScript (`.pt`) | torch | `torch.jit.load` 推理（CPU/CUDA） |
| [python_paddle](python_paddle/) | PaddlePaddle (`.pdmodel` / `.pdiparams`) | paddlepaddle | Paddle Inference 推理 |
| [python_openvino](python_openvino/) | OpenVINO IR (`.xml` / `.bin`) | openvino | OpenVINO Core 推理 |
| [python_tensorrt](python_tensorrt/) | TensorRT (`.engine`) | tensorrt + pycuda | TRT 引擎反序列化推理 |
| [python_caffe](python_caffe/) | Caffe (`.prototxt` + `.caffemodel`) | opencv-python | OpenCV DNN 读 Caffe |
| [python_gguf](python_gguf/) | GGUF (`.gguf`) | llama-cpp-python | 大语言模型（LLM）对话/生成 |
| [python_jax](python_jax/) | JAX / Flax checkpoint | jax + flax | JAX/Flax 参数加载推理 |

### C++

| 目录 | 模型格式 | 运行库/框架 | 说明 |
|------|----------|-------------|------|
| [cpp_onnxruntime](cpp_onnxruntime/) | ONNX (`.onnx`) | ONNX Runtime C++ | 原生 ONNX Runtime 会话推理 |
| [cpp_opencv_dnn](cpp_opencv_dnn/) | ONNX / Caffe / TF / Darknet | OpenCV DNN | OpenCV `readNet` 通用推理 |
| [cpp_tensorrt](cpp_tensorrt/) | TensorRT (`.engine`) | TensorRT + CUDA | TRT 引擎反序列化与执行上下文 |
| [cpp_libtorch](cpp_libtorch/) | TorchScript (`.pt`) | LibTorch | C++ 版 PyTorch 推理 |
| [cpp_openvino](cpp_openvino/) | OpenVINO IR (`.xml` / `.bin`) | OpenVINO C++ | OpenVINO 编译模型推理 |
| [cpp_tflite](cpp_tflite/) | TensorFlow Lite (`.tflite`) | TFLite C++ | TFLite 解释器推理 |

### Go

| 目录 | 模型格式 | 运行库/框架 | 说明 |
|------|----------|-------------|------|
| [go_onnxruntime](go_onnxruntime/) | ONNX (`.onnx`) | onnxruntime_go | Go 版 ONNX Runtime |
| [go_tflite](go_tflite/) | TensorFlow Lite (`.tflite`) | go-tflite (+XNNPACK) | Go 版 TFLite 推理 |
| [go_gguf](go_gguf/) | GGUF (`.gguf`) | llama.cpp (llama-cli) | Go 调用 llama.cpp 子进程做 LLM 推理 |

### Rust

| 目录 | 模型格式 | 运行库/框架 | 说明 |
|------|----------|-------------|------|
| [rust_tract](rust_tract/) | ONNX / TF (`.onnx` 等) | tract-onnx | tract 轻量推理框架 |
| [rust_candle](rust_candle/) | safetensors / `.pt` | candle-core | Candle（PyTorch 风格）推理 |
| [rust_ort](rust_ort/) | ONNX (`.onnx`) | ort | ONNX Runtime Rust 绑定 |

### Java

| 目录 | 模型格式 | 运行库/框架 | 说明 |
|------|----------|-------------|------|
| [java_onnxruntime](java_onnxruntime/) | ONNX (`.onnx`) | onnxruntime-java | Java 版 ONNX Runtime |
| [java_djl](java_djl/) | 多格式（MXNet/PyTorch/ONNX 等） | Deep Java Library | DJL 统一模型加载与翻译器 |

### C# (.NET)

| 目录 | 模型格式 | 运行库/框架 | 说明 |
|------|----------|-------------|------|
| [csharp_onnxruntime](csharp_onnxruntime/) | ONNX (`.onnx`) | Microsoft.ML.OnnxRuntime | .NET 版 ONNX Runtime |

### JavaScript / Node.js

| 目录 | 模型格式 | 运行库/框架 | 说明 |
|------|----------|-------------|------|
| [node_onnxruntime](node_onnxruntime/) | ONNX (`.onnx`) | onnxruntime-node | Node.js 版 ONNX Runtime |
| [node_tfjs](node_tfjs/) | TF SavedModel / TF.js | @tensorflow/tfjs-node | TensorFlow.js 推理 |

## 可验证性矩阵（machine-readable）

仓库根的 `matrix.json` 记录每个示例的真实可验证性状态，由脚本从源码提取生成，**不是手写**：

```
python tools/gen_matrix.py          # 重新提取并覆盖 matrix.json
python tools/gen_matrix.py --check  # 校验 matrix.json 与源码一致，不一致退出码 1
```

`--check` 只校验**与工具链无关的源码事实**（未消除的 TODO、推理调用是否存在、模型格式、
入口文件、缺失依赖清单），因此可以在 CI 里稳定运行。它有意不校验 `status` / `compile_check` /
`evidence` 字段——这三个字段依赖本机是否装了 g++ / node / gofmt / dotnet / cargo 与 ONNX Runtime
共享库，跨机器比对会产生假失败。想确认状态字段是否最新，跑一次
`python tools/gen_matrix.py && PYTHON_BIN=<你的 venv python> python tools/verify_matrix.py`。

状态定义（三选一，判定规则见 `tools/gen_matrix.py` 顶部 docstring）：

| 状态 | 含义 |
|------|------|
| `verified` | 由 `tools/verify_matrix.py` 真实构建并运行通过，命令与输出记录在 `matrix.json` 的 `evidence` 字段 |
| `compilable` | 语法/编译检查通过（Python `py_compile` 等价语法检查、Go `gofmt -l`、JS `node --check`、C++ `g++ -fsyntax-only`），但推理调用尚未执行（缺模型或运行库） |
| `placeholder` | 存在未消除 `TODO`，或推理调用缺失/被注释掉，或语法编译检查失败 |

`verified` 用例的复现方式（依赖装在示例外部的虚拟环境，不污染仓库）：

```bash
python -m venv .venv
.venv/Scripts/pip install -r python_onnx/requirements.txt \
                          -r python_pb/requirements.txt \
                          -r python_pth_pt/requirements.txt
PYTHON_BIN=.venv/Scripts/python.exe python tools/verify_matrix.py

# go_onnxruntime 需要 onnxruntime 的 C 共享库（pip 安装的 onnxruntime 包内含 capi/onnxruntime.dll）
# node_onnxruntime 需要 node_onnxruntime/node_modules（npm ci）
```

当前 `verified` 的示例与实跑证据（模型与输入图片均随仓库提供，不依赖外网）：

| 示例 | 实际执行的推理 |
|------|----------------|
| `python_onnx` | `yolov5n.onnx` + `example2.png`，onnxruntime 10 次计时 + NMS + 画框 |
| `python_pb` | `yolov3_resnet18.pb`（tf.compat.v1 冻结图）+ `example2.png`，TensorFlow 2.22 跑 10 次 |
| `python_pth_pt` | `yolov5s.pt` + `example.png`，torch 2.15 前向 + numpy NMS，输出 9 个检测框 |
| `go_onnxruntime` | `go build` 后链接 `onnxruntime.dll`，`yolov5n.onnx` 推理并打印输出 shape |
| `node_onnxruntime` | `onnxruntime-node` 加载 `yolov5n.onnx` 推理并打印输出 shape |

> 本仓库**不包含** NPU 设备与厂商 SDK（CANN / RKNPU2 / SNPE / QNN / NeuroPilot / Core ML 等），
> 其余示例只保证语法与代码结构正确，真实加速卡推理需要相应硬件与工具链，
> 详见 `matrix.json` 的 `missing_deps` 字段与 `signals`（未消除 TODO 行号、被注释掉的推理调用、
> `return []` 空壳函数）。

## 使用方式

1. 进入对应子目录，查看该语言/框架的 `main` 文件与依赖清单。
2. 将 `TODO` 占位处替换为实际模型路径、输入输出名、输入尺寸、类别数等动态参数。
3. 准备输入图片与模型文件后，按各目录的构建/运行说明执行（Python `requirements.txt`、C++ `CMakeLists.txt`、Go `go.mod`、Rust `Cargo.toml`、Java `pom.xml`、C# `.csproj`、Node `package.json`）。
