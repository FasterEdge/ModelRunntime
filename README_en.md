<div align="center">
  <img src="https://avatars.githubusercontent.com/u/245985800?s=200&v=4" alt="logo" width="100" />
  <h2>ModelRunntime</h2>
  <h3>Model Inference Runtime Examples Across Languages and Formats</h3>
</div>

### 1. Introduction

- ModelRunntime is a collection of model inference runtime examples covering multiple programming languages and model formats.
- **Each subdirectory is an independent runtime example.** All model-specific dynamic parameters—including model paths, input and output tensor names, input dimensions, class counts, quantization parameters, and devices—are represented by `TODO` comments and must be filled in for the actual model before use.
- Input images such as `example.png` and `example2.png`, together with the required model files, must be prepared by the user where they are not already present.

### 2. Python Examples

| Directory | Model format | Runtime / framework | Description |
|-----------|--------------|---------------------|-------------|
| [python_onnx](python_onnx/) | ONNX (`.onnx`) | onnxruntime | Complete YOLOv5n object-detection example with NMS and visualization |
| [python_pb](python_pb/) | TensorFlow (`.pb`) | tensorflow>=2.16 | YOLOv3-ResNet18 object detection (tf.compat.v1 frozen graph) with a Dockerfile |
| [python_pth_pt](python_pth_pt/) | PyTorch (`.pt` / `.pth`) | torch | YOLOv5s weight-loading example |
| [python_tflite](python_tflite/) | TensorFlow Lite (`.tflite`) | tensorflow | TFLite interpreter inference with quantization notes |
| [python_tf_savedmodel](python_tf_savedmodel/) | TF2 SavedModel | tensorflow | SavedModel signature inference |
| [python_keras_h5](python_keras_h5/) | Keras (`.h5` / `.keras`) | keras | Keras model loading and prediction |
| [python_torchscript](python_torchscript/) | TorchScript (`.pt`) | torch | `torch.jit.load` inference on CPU or CUDA |
| [python_paddle](python_paddle/) | PaddlePaddle (`.pdmodel` / `.pdiparams`) | paddlepaddle | Paddle Inference example |
| [python_openvino](python_openvino/) | OpenVINO IR (`.xml` / `.bin`) | openvino | OpenVINO Core inference |
| [python_tensorrt](python_tensorrt/) | TensorRT (`.engine`) | tensorrt + pycuda | TensorRT engine deserialization and inference |
| [python_caffe](python_caffe/) | Caffe (`.prototxt` + `.caffemodel`) | opencv-python | Loading Caffe through OpenCV DNN |
| [python_gguf](python_gguf/) | GGUF (`.gguf`) | llama-cpp-python | Large language model conversation and generation |
| [python_jax](python_jax/) | JAX / Flax checkpoint | jax + flax | JAX / Flax parameter loading and inference |

### 3. C++ Examples

| Directory | Model format | Runtime / framework | Description |
|-----------|--------------|---------------------|-------------|
| [cpp_onnxruntime](cpp_onnxruntime/) | ONNX (`.onnx`) | ONNX Runtime C++ | Native ONNX Runtime session inference |
| [cpp_opencv_dnn](cpp_opencv_dnn/) | ONNX / Caffe / TF / Darknet | OpenCV DNN | General inference through OpenCV `readNet` |
| [cpp_tensorrt](cpp_tensorrt/) | TensorRT (`.engine`) | TensorRT + CUDA | TensorRT engine deserialization and execution context |
| [cpp_libtorch](cpp_libtorch/) | TorchScript (`.pt`) | LibTorch | PyTorch inference in C++ |
| [cpp_openvino](cpp_openvino/) | OpenVINO IR (`.xml` / `.bin`) | OpenVINO C++ | OpenVINO compiled-model inference |
| [cpp_tflite](cpp_tflite/) | TensorFlow Lite (`.tflite`) | TFLite C++ | TFLite interpreter inference |

### 4. Go Examples

| Directory | Model format | Runtime / framework | Description |
|-----------|--------------|---------------------|-------------|
| [go_onnxruntime](go_onnxruntime/) | ONNX (`.onnx`) | onnxruntime_go | ONNX Runtime in Go |
| [go_tflite](go_tflite/) | TensorFlow Lite (`.tflite`) | go-tflite (+XNNPACK) | TFLite inference in Go |
| [go_gguf](go_gguf/) | GGUF (`.gguf`) | llama.cpp (`llama-cli`) | Go invokes a llama.cpp subprocess for LLM inference |

### 5. Rust Examples

| Directory | Model format | Runtime / framework | Description |
|-----------|--------------|---------------------|-------------|
| [rust_tract](rust_tract/) | ONNX / TF (`.onnx`, etc.) | tract-onnx | Lightweight inference with tract |
| [rust_candle](rust_candle/) | safetensors / `.pt` | candle-core | PyTorch-style inference with Candle |
| [rust_ort](rust_ort/) | ONNX (`.onnx`) | ort | ONNX Runtime Rust bindings |

### 6. Java Examples

| Directory | Model format | Runtime / framework | Description |
|-----------|--------------|---------------------|-------------|
| [java_onnxruntime](java_onnxruntime/) | ONNX (`.onnx`) | onnxruntime-java | ONNX Runtime in Java |
| [java_djl](java_djl/) | Multiple formats (MXNet / PyTorch / ONNX, etc.) | Deep Java Library | Unified DJL model loading and translators |

### 7. C# (.NET) Example

| Directory | Model format | Runtime / framework | Description |
|-----------|--------------|---------------------|-------------|
| [csharp_onnxruntime](csharp_onnxruntime/) | ONNX (`.onnx`) | Microsoft.ML.OnnxRuntime | ONNX Runtime in .NET |

### 8. JavaScript / Node.js Examples

| Directory | Model format | Runtime / framework | Description |
|-----------|--------------|---------------------|-------------|
| [node_onnxruntime](node_onnxruntime/) | ONNX (`.onnx`) | onnxruntime-node | ONNX Runtime in Node.js |
| [node_tfjs](node_tfjs/) | TF SavedModel / TF.js | @tensorflow/tfjs-node | TensorFlow.js inference |

### 9. Verifiability Matrix (machine-readable)

`matrix.json` at the repository root records the real verifiability status of every example. It is
generated by a script from the source code — **it is never hand-written**:

```
python tools/gen_matrix.py          # re-extract and overwrite matrix.json
python tools/gen_matrix.py --check  # verify matrix.json matches the source; exit code 1 on drift
```

`--check` only compares **source facts that do not depend on the local toolchain** (unresolved TODOs,
whether an inference call exists, model formats, entry file, bundled model files, missing-dependency
list), so it runs stably in CI. It deliberately excludes `status` / `compile_check` / `evidence`
(these depend on whether g++ / node / gofmt / dotnet / cargo and the ONNX Runtime shared library are
installed) and also excludes `environment` (which records the local Python version, so a different
Python patch release would not cause a false failure). To confirm the status fields are current, run
`python tools/gen_matrix.py && PYTHON_BIN=<your venv python> python tools/verify_matrix.py`.

`tools/verify_matrix.py` has its own `--check` with a different meaning: it is **read-only**. It
checks that every example marked `verified` in `matrix.json` carries a valid `evidence` block, and
returns that verdict as the exit code without re-running anything or rewriting the file. A
"check" step in CI therefore cannot silently degrade into "rebuild and overwrite the evidence".

Absolute local paths inside `evidence.command` are replaced with placeholders (`<repo>/`,
`<python>`, `<tmp>`, `<home>`) when the evidence is written back, keeping only the information that
matters (“which interpreter, which temp directory”). The matrix is committed, so baking in
`C:\Users\<user>\...` would make it impossible for anyone else to reproduce the command and would
leak local details into the repository. `evidence.output_tail` is the tail of the output actually
produced by that run; non-ASCII text is decoded strictly as UTF-8 or the local code page, so no
U+FFFD ever lands in the matrix.

Static metadata (platform, runtime, missing dependencies) lives in `tools/matrix_spec.json`;
`gen_matrix.py` reads it and cross-checks it against what the source actually contains.

Status definitions (exactly one per example; the rules are in the docstring at the top of
`tools/gen_matrix.py`):

| Status | Meaning |
|--------|---------|
| `verified` | Really built and run to success by `tools/verify_matrix.py`; the command and output are recorded in the `evidence` field of `matrix.json` |
| `compilable` | Syntax/compile check passed (Python `py_compile`-equivalent, Go `gofmt -l`, JS `node --check`, C++ `g++ -fsyntax-only`, Java `javac -cp <local m2 jars>`, C# `dotnet build`, Rust `cargo check --offline`), but inference has not been executed (model or runtime library missing) |
| `placeholder` | An unresolved `TODO` exists, the inference call is missing or commented out, or the compile check failed |

The `compile_check` field records the real result per language. Measured on this machine (2026-11-23):

- Python / Go / JS: all `ok=true` (`ast.parse`, `gofmt -l`, `node --check`).
- Java: `javac -cp <local m2: onnxruntime-1.32.0.jar / djl api-0.40.0.jar>` really compiles, `ok=true`;
  both Java examples are still classified placeholder because of unresolved `TODO`s.
- Rust: `cargo check --offline` passes under the GNU toolchain (`rust_tract`, `rust_candle`);
  `rust_ort` fails because this machine lacks `link.exe` and `ort-sys` ships no windows-gnu
  prebuilt binary, recorded as `ok=false` + `skip=true` (missing environment, not a syntax error).
- C#: no .NET SDK, recorded as `ok=false` + `skip=true`.
- C++: `g++ -fsyntax-only` fails on missing third-party headers (ONNX Runtime / OpenCV / TensorRT,
  …); that is exactly the evidence for `missing_deps` and is not marked as `skip`.

Measured on the same source tree: `compile_check` is `ok=true` for 22 examples
(13 Python + 3 Go + 2 Rust + 2 Java + 2 JS), `skip=true` for 2 (`rust_ort` missing `link.exe` /
no windows-gnu prebuilt, `csharp_onnxruntime` no .NET SDK), and 6 C++ examples fail on missing
third-party headers (the evidence for `missing_deps`). Read the `summary` and `compile_check` of
the run you actually made rather than copying these numbers.

How to reproduce the `verified` cases (dependencies go into a virtual environment outside the
examples, so the repository stays clean):

```bash
python -m venv .venv
.venv/Scripts/pip install -r python_onnx/requirements.txt \
                          -r python_pb/requirements.txt \
                          -r python_pth_pt/requirements.txt
PYTHON_BIN=.venv/Scripts/python.exe python tools/verify_matrix.py

# go_onnxruntime needs the ONNX Runtime C shared library
#   (the pip onnxruntime package ships capi/onnxruntime.dll)
# node_onnxruntime needs node_onnxruntime/node_modules (npm ci)
```

The examples currently `verified`, with the inference that actually ran (models and input images are
shipped in the repository; no external network needed):

| Example | Inference actually executed |
|---------|-----------------------------|
| `python_onnx` | `yolov5n.onnx` + `example2.png`: onnxruntime, 10 timed runs + NMS + box-drawing steps (on this machine it detects 1 target and writes `result_example2.png`, but draws 0 boxes) |
| `python_pb` | `yolov3_resnet18.pb` (tf.compat.v1 frozen graph) + `example2.png`: TensorFlow **2.22.0-rc1 (pre-release)** 10 runs |
| `python_pth_pt` | `yolov5s.pt` + `example.png`: torch 2.15 forward + numpy NMS, 9 detections printed |
| `go_onnxruntime` | `go build` linked against `onnxruntime.dll`, `yolov5n.onnx` inference, prints output shape |
| `node_onnxruntime` | `onnxruntime-node` loads `yolov5n.onnx`, runs inference and prints output shape |

> On the TensorFlow version for `python_pb`: `python_pb/requirements.txt` only pins
> `tensorflow>=2.21.0`, and the environment this was measured in resolved to **2.22.0-rc1** (`-rc1` is
> a PyPI pre-release, not the stable 2.22.0). For a stable release, pin `tensorflow==2.22.0` or `>=2.23`.

### 10. Matrix size

The `summary` field of `matrix.json` is measured by the script on every run. Measured on 2026-11-23
(Python 3.14.7): **total=30**; running only `gen_matrix.py` gives
`verified=0 / compilable=1 / placeholder=29`, and after also running `verify_matrix.py` it becomes
`verified=5 / compilable=0 / placeholder=25`. These counts change whenever the source or the local
toolchain changes — read the `summary` in `matrix.json` of that run rather than copying this line.
Static metadata source: `tools/matrix_spec.json` (platform, runtime and
missing-dependency list per example). The `bundled_models` field lists model files that actually exist
in the example directory and are shipped with the repository (case-insensitive match against
`FORMAT_SUFFIXES`, with `node_modules` / `target` / `bin` / `obj` / `dist` / `__pycache__` and other
artifact directories excluded); an empty field means the model must be supplied before running.

> This repository **does not include** NPU devices or vendor SDKs (CANN / RKNPU2 / SNPE / QNN /
> NeuroPilot / Core ML, etc.). The remaining examples are only guaranteed to be syntactically and
> structurally correct; real accelerator inference needs the corresponding hardware and toolchain.
> See `missing_deps` in `matrix.json` and the `signals` field (unresolved TODO line numbers,
> commented-out inference calls, `return []` shell functions).

### 11. Usage

1. Enter the relevant subdirectory and inspect the language or framework's `main` file and dependency manifest.
2. Replace every `TODO` placeholder with the actual model path, input and output names, input dimensions, class count, and other model-specific dynamic parameters.
3. Prepare the input images and model files, then build or run the example using its manifest: Python `requirements.txt`, C++ `CMakeLists.txt`, Go `go.mod`, Rust `Cargo.toml`, Java `pom.xml`, C# `.csproj`, or Node.js `package.json`.
