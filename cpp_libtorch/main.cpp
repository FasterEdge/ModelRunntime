// LibTorch (C++ PyTorch) 推理示例
// 动态参数均用注释占位，按实际模型修改

#include <iostream>
#include <chrono>
#include <torch/script.h>
#include <torch/torch.h>

int main() {
    // ===== 动态参数（按实际模型修改）=====
    // MODEL_PATH: TorchScript .pt 模型路径
    std::string model_path = "model_scripted.pt";  // TODO: 替换为 .pt 模型路径
    // INPUT_SHAPE: 输入张量 shape {N,C,H,W}
    const int batch = 1, channels = 3, height = 640, width = 640;  // TODO: 按模型修改
    // DEVICE: "cpu" / "cuda"
    torch::Device device(torch::kCPU);             // TODO: 按可用设备修改

    torch::jit::script::Module module;
    try {
        module = torch::jit::load(model_path, device);
    } catch (const c10::Error& e) {
        std::cerr << "failed to load model: " << e.what() << std::endl;
        return -1;
    }
    module.eval();

    // TODO: 填充实际输入数据（读图、预处理 -> CHW）
    auto input_tensor = torch::zeros({batch, channels, height, width}, torch::kFloat32).to(device);

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input_tensor);

    // 预热
    for (int i = 0; i < 3; i++) {
        auto out = module.forward(inputs);
        (void)out;
    }

    auto start = std::chrono::steady_clock::now();
    auto output = module.forward(inputs);
    auto end = std::chrono::steady_clock::now();
    std::cout << "inference time: "
              << std::chrono::duration<double, std::milli>(end - start).count()
              << " ms" << std::endl;

    // TODO: 解析输出张量（output.toTensor()）
    return 0;
}
