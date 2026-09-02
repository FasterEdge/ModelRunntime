// FasterEdge 开源项目 - Github: https://github.com/FasterEdge - Gitee: https://gitee.com/FasterEdge
// OpenVINO C++ 推理示例
// 动态参数均用注释占位，按实际模型修改

#include <iostream>
#include <chrono>
#include <vector>
#include <openvino/openvino.hpp>

int main() {
    // ===== 动态参数（按实际模型修改）=====
    // MODEL_PATH: OpenVINO IR .xml 模型路径
    std::string model_path = "model.xml";     // TODO: 替换为 .xml 模型路径
    // DEVICE: "CPU" / "GPU" / "AUTO"
    std::string device = "CPU";               // TODO: 按可用设备修改
    // INPUT_SHAPE: 输入张量 shape
    const int batch = 1, channels = 3, height = 640, width = 640;  // TODO: 按模型修改

    ov::Core core;
    std::shared_ptr<ov::Model> model = core.read_model(model_path);
    ov::CompiledModel compiled_model = core.compile_model(model, device);

    ov::InferRequest infer_request = compiled_model.create_infer_request();

    // TODO: 填充实际输入数据（读图、预处理 -> CHW）
    std::vector<float> input_data(batch * channels * height * width, 0.f);

    // TODO: 输入张量名按模型查询（model->input(0)）
    ov::Tensor input_tensor(ov::element::f32, {batch, channels, height, width},
                            input_data.data());
    infer_request.set_input_tensor(input_tensor);

    // 预热
    for (int i = 0; i < 3; i++) infer_request.infer();

    auto start = std::chrono::steady_clock::now();
    infer_request.infer();
    auto end = std::chrono::steady_clock::now();
    std::cout << "inference time: "
              << std::chrono::duration<double, std::milli>(end - start).count()
              << " ms" << std::endl;

    // TODO: 读取输出（infer_request.get_output_tensor(0)）并后处理
    return 0;
}
