// TensorFlow Lite C++ 推理示例
// 动态参数均用注释占位，按实际模型修改

#include <iostream>
#include <chrono>
#include <vector>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

int main() {
    // ===== 动态参数（按实际模型修改）=====
    // MODEL_PATH: .tflite 模型路径
    std::string model_path = "model.tflite";   // TODO: 替换为 .tflite 模型路径

    std::unique_ptr<tflite::FlatBufferModel> model =
        tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    if (!model) {
        std::cerr << "failed to load model" << std::endl;
        return -1;
    }

    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);
    if (!interpreter) return -1;

    interpreter->AllocateTensors();

    int input_index = interpreter->inputs()[0];      // TODO: 按模型输入索引修改
    int output_index = interpreter->outputs()[0];    // TODO: 按模型输出索引修改

    // TODO: 填充输入数据（注意量化模型需按 scale/zero_point 转换）
    // float* input = interpreter->typed_input_tensor<float>(0);

    // 预热
    for (int i = 0; i < 3; i++) interpreter->Invoke();

    auto start = std::chrono::steady_clock::now();
    interpreter->Invoke();
    auto end = std::chrono::steady_clock::now();
    std::cout << "inference time: "
              << std::chrono::duration<double, std::milli>(end - start).count()
              << " ms" << std::endl;

    // TODO: 读取输出并后处理
    return 0;
}
