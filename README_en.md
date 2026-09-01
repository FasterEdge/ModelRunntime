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
| [python_pb](python_pb/) | TensorFlow 1.x (`.pb`) | tensorflow==1.15.4 | YOLOv3-ResNet18 object detection with a Dockerfile |
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

### 9. Usage

1. Enter the relevant subdirectory and inspect the language or framework's `main` file and dependency manifest.
2. Replace every `TODO` placeholder with the actual model path, input and output names, input dimensions, class count, and other model-specific dynamic parameters.
3. Prepare the input images and model files, then build or run the example using its manifest: Python `requirements.txt`, C++ `CMakeLists.txt`, Go `go.mod`, Rust `Cargo.toml`, Java `pom.xml`, C# `.csproj`, or Node.js `package.json`.
