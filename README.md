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
入口文件、随仓库分发的模型文件、缺失依赖清单），因此可以在 CI 里稳定运行。它有意不校验
`status` / `compile_check` / `evidence` 字段——这三个字段依赖本机是否装了 g++ / node /
gofmt / dotnet / cargo 与 ONNX Runtime 共享库，跨机器比对会产生假失败；同理
`environment`（记录本机 Python 版本）也不参与比对，换 Python 小版本不会造成假失败。
想确认状态字段是否最新，跑一次
`python tools/gen_matrix.py && PYTHON_BIN=<你的 venv python> python tools/verify_matrix.py`。

`tools/verify_matrix.py` 也有一个 `--check`，语义与前一个不同：它**只读**，检查 `matrix.json`
里已标记 `verified` 的用例是否都带有有效 `evidence`，不重跑、不改文件、退出码即结论。
这样 CI 里的“检查”不会退化成“重建并顺手覆盖证据”。

`evidence.command` 里的本机绝对路径在写回时会被替换成占位符（`<repo>/`、
`<python>`、`<tmp>`、`<home>`），只保留“用哪个解释器、在哪个临时目录跑”这层信息。
矩阵随仓库提交，写死 `C:\Users\<用户名>\...` 会让别人无法照着重跑，也把本机信息带进了
版本库；`evidence.output_tail` 是当场实测输出的尾部，中文按 UTF-8 或本地代码页严格解码，
不会有 U+FFFD。

矩阵的静态元数据（平台、运行库、缺失依赖清单）集中在 `tools/matrix_spec.json`，
由 `gen_matrix.py` 读取后与源码实况交叉生成；示例总数与状态计数写在 `matrix.json` 的
`summary` 字段，可随每次重新生成实测更新。

状态定义（三选一，判定规则见 `tools/gen_matrix.py` 顶部 docstring）：

| 状态 | 含义 |
|------|------|
| `verified` | 由 `tools/verify_matrix.py` 真实构建并运行通过，命令与输出记录在 `matrix.json` 的 `evidence` 字段 |
| `compilable` | 语法/编译检查通过（Python `py_compile` 等价语法检查、Go `gofmt -l`、JS `node --check`、C++ `g++ -fsyntax-only`、Java `javac -cp <本地 m2 jar>`、C# `dotnet build`、Rust `cargo check --offline`），但推理调用尚未执行（缺模型或运行库） |
| `placeholder` | 存在未消除 `TODO`，或推理调用缺失/被注释掉，或语法编译检查失败 |

`compile_check` 字段如实记录每类语言的检查结果。本机（2026-11-23）实测：

- Python / Go / JS：全部 `ok=true`（`ast.parse`、`gofmt -l`、`node --check`）。
- Java：`javac -cp <本地 m2 的 onnxruntime-1.32.0.jar / djl api-0.40.0.jar>` 真实编译通过，`ok=true`；
  两个 Java 示例仍因未消除 `TODO` 而判为 placeholder。
- Rust：`cargo check --offline` 在 GNU toolchain 下通过（`rust_tract` / `rust_candle`）；
  `rust_ort` 因本机缺 `link.exe` 且 `ort-sys` 无 windows-gnu 预编译产物而失败，
  记为 `ok=false` + `skip=true`（环境缺失，不是源码语法错）。
- C#：无 .NET SDK，记 `ok=false` + `skip=true`。
- C++：`g++ -fsyntax-only` 因缺第三方头文件（ONNX Runtime / OpenCV / TensorRT 等）失败，
  这正是 `missing_deps` 的证据，不加 `skip` 标记。

即：`ok=false` 有两类原因——本机工具链缺失（`skip=true`）与第三方 SDK 头文件缺失（C++，`skip` 不设），
都不是「源码语法错误」。

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
| `python_onnx` | `yolov5n.onnx` + `example2.png`，onnxruntime 10 次计时 + NMS + 画框步骤（本机实跑检测到 1 个目标、写出 `result_example2.png`，但绘制框数为 0） |
| `python_pb` | `yolov3_resnet18.pb`（tf.compat.v1 冻结图）+ `example2.png`，TensorFlow **2.22.0-rc1（预发布版）** 跑 10 次 |
| `python_pth_pt` | `yolov5s.pt` + `example.png`，torch 2.15 前向 + numpy NMS，输出 9 个检测框 |
| `go_onnxruntime` | `go build` 后链接 `onnxruntime.dll`，`yolov5n.onnx` 推理并打印输出 shape |
| `node_onnxruntime` | `onnxruntime-node` 加载 `yolov5n.onnx` 推理并打印输出 shape |

> 关于 `python_pb` 的 TensorFlow 版本：`python_pb/requirements.txt` 只写了 `tensorflow>=2.21.0`，
> 本机实跑环境解析到的是 **2.22.0-rc1**（`-rc1` 是 PyPI 上的预发布版本号），并非稳定版 2.22.0。
> 若需稳定版，请在 requirements 里 pin `tensorflow==2.22.0` 或 `>=2.23`。

### 矩阵规模

`matrix.json` 的 `summary` 字段由脚本实测填写。本次实测（2026-11-23，Python 3.14.7）：
**total=30**；只跑 `gen_matrix.py` 时为 `verified=0 / compilable=1 / placeholder=29`，
再跑 `verify_matrix.py` 后变为 `verified=5 / compilable=0 / placeholder=25`。
同一份源码的 `compile_check` 实测 `ok=true` 共 22 个（13 Python + 3 Go + 2 Rust + 2 Java + 2 JS），
`skip=true` 2 个（`rust_ort` 缺 `link.exe`/无 windows-gnu 预编译、`csharp_onnxruntime` 无 .NET SDK），
6 个 C++ 因缺第三方头文件失败（即 `missing_deps` 的证据）。
数字会随源码与工具链变化，请以当次 `matrix.json` 的 `summary` 与 `compile_check` 为准，不要照抄本行。
静态元数据来源：`tools/matrix_spec.json`（每个示例的平台、运行库、缺失依赖清单）。
`bundled_models` 字段列出示例目录内实际存在、随仓库分发的模型文件（按 `FORMAT_SUFFIXES`
大小写无关匹配，并排除 `node_modules` / `target` / `bin` / `obj` / `dist` / `__pycache__` 等产物目录）；
该字段为空表示运行前须自行准备模型。

> 本仓库**不包含** NPU 设备与厂商 SDK（CANN / RKNPU2 / SNPE / QNN / NeuroPilot / Core ML 等），
> 其余示例只保证语法与代码结构正确，真实加速卡推理需要相应硬件与工具链，
> 详见 `matrix.json` 的 `missing_deps` 字段与 `signals`（未消除 TODO 行号、被注释掉的推理调用、
> `return []` 空壳函数）。

## 使用方式

1. 进入对应子目录，查看该语言/框架的 `main` 文件与依赖清单。
2. 将 `TODO` 占位处替换为实际模型路径、输入输出名、输入尺寸、类别数等动态参数。
3. 准备输入图片与模型文件后，按各目录的构建/运行说明执行（Python `requirements.txt`、C++ `CMakeLists.txt`、Go `go.mod`、Rust `Cargo.toml`、Java `pom.xml`、C# `.csproj`、Node `package.json`）。
