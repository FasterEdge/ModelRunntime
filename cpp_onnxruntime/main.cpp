// FasterEdge 开源项目 - Github: https://github.com/FasterEdge - Gitee: https://gitee.com/FasterEdge
// ONNX Runtime C++ 推理示例
// 动态参数均用注释占位，按实际模型修改

#include <iostream>
#include <chrono>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>

int main() {
    // ===== 动态参数（按实际模型修改）=====
    // MODEL_PATH: .onnx 模型路径
    const char* model_path = "model.onnx";   // TODO: 替换为你的 .onnx 模型路径
    // INPUT_SHAPE: 输入张量 shape，{N,C,H,W} 或 {N,H,W,C}
    const std::vector<int64_t> input_shape = {1, 3, 640, 640};  // TODO: 按模型输入修改
    // INPUT_NAME / OUTPUT_NAME: 张量名（可用 session.GetInputNameAllocated 查询）
    const char* input_name = "images";       // TODO: 按模型输入名修改
    const char* output_name = "output0";     // TODO: 按模型输出名修改

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "cpp_onnxruntime");
    Ort::SessionOptions session_options;
    // session_options.SetIntraOpNumThreads(4);  // TODO: 按 CPU 核数调整

    Ort::Session session(env, model_path, session_options);

    auto allocator_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    // TODO: 填充实际输入数据（读图、预处理）
    std::vector<float> input_data(1 * 3 * 640 * 640, 0.f);

    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(Ort::Value::CreateTensor<float>(
        allocator_info, input_data.data(), input_data.size(), input_shape.data(), input_shape.size()));

    const char* input_names[] = {input_name};
    const char* output_names[] = {output_name};

    // 预热
    for (int i = 0; i < 3; i++)
        session.Run(Ort::RunOptions{nullptr}, input_names, input_tensors.data(), 1, output_names, 1);

    // 计时推理
    auto start = std::chrono::steady_clock::now();
    auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names, input_tensors.data(), 1, output_names, 1);
    auto end = std::chrono::steady_clock::now();
    std::cout << "inference time: "
              << std::chrono::duration<double, std::milli>(end - start).count()
              << " ms" << std::endl;

    // TODO: 读取输出张量数据并后处理
    return 0;
}
