# ModelRunntime

多种语言、多种格式的模型推理运行时示例集。

> 说明：每个子目录是一个独立的模型运行时示例。**所有与模型强相关的动态参数（模型路径、输入/输出张量名、输入尺寸、类别数、量化参数、设备等）均以 `TODO` 注释占位**，请按实际模型填写后使用。示例中的 `example.png` / `example2.png` 等输入图片与模型文件需自行准备。

## 目录索引

### Python

| 目录 | 模型格式 | 运行库/框架 | 说明 |
|------|----------|-------------|------|
| [python_onnx](python_onnx/) | ONNX (`.onnx`) | onnxruntime | YOLOv5n 目标检测完整示例（含 NMS 与可视化） |
| [python_pb](python_pb/) | TensorFlow 1.x (`.pb`) | tensorflow==1.15.4 | YOLOv3-ResNet18 目标检测，附 Dockerfile |
| [python_pth_pt](python_pth_pt/) | PyTorch (`.pt` / `.pth`) | torch | YOLOv5s 权重加载示例 |
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

## 使用方式

1. 进入对应子目录，查看该语言/框架的 `main` 文件与依赖清单。
2. 将 `TODO` 占位处替换为实际模型路径、输入输出名、输入尺寸、类别数等动态参数。
3. 准备输入图片与模型文件后，按各目录的构建/运行说明执行（Python `requirements.txt`、C++ `CMakeLists.txt`、Go `go.mod`、Rust `Cargo.toml`、Java `pom.xml`、C# `.csproj`、Node `package.json`）。
