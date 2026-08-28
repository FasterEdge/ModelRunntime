// TensorRT C++ 推理示例
// 动态参数均用注释占位，按实际模型修改

#include <iostream>
#include <chrono>
#include <vector>
#include <cuda_runtime.h>
#include <NvInfer.h>
#include <NvOnnxParser.h>

// 简单 RAII 封装日志器（完整实现略）
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) std::cout << "[TRT] " << msg << std::endl;
    }
};

int main() {
    // ===== 动态参数（按实际模型修改）=====
    // ENGINE_PATH: 已构建好的 .engine 文件路径
    const char* engine_path = "model.engine";   // TODO: 替换为 .engine 路径
    // INPUT_SHAPE: 输入张量 shape
    const int batch_size = 1, channels = 3, height = 640, width = 640;  // TODO: 按模型修改

    Logger logger;
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger);

    // 读取 engine 文件
    std::ifstream file(engine_path, std::ios::binary);
    std::vector<char> buffer((std::istreambuf_iterator<char>(file)),
                             std::istreambuf_iterator<char>());
    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(buffer.data(), buffer.size());
    nvinfer1::IExecutionContext* context = engine->createExecutionContext();

    // TODO: 分配 host/device 内存、绑定 bindings（此处用简单占位）
    // 可通过 engine->getTensorName(i) 查询输入/输出名
    // context->setInputShape(...);  // TODO: 动态 shape 需设置

    // TODO: 填充输入数据并 cudaMemcpy 到 device
    // TODO: context->enqueueV3(...) 执行推理
    // TODO: cudaMemcpy 回 host 并后处理

    std::cout << "engine loaded, ready for inference (fill in bindings)" << std::endl;

    context->destroy();
    engine->destroy();
    runtime->destroy();
    return 0;
}
