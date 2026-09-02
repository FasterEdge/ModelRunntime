// FasterEdge 开源项目 - Github: https://github.com/FasterEdge - Gitee: https://gitee.com/FasterEdge
// OpenCV DNN 推理示例（支持 ONNX / Caffe / TensorFlow / Darknet 等）
// 动态参数均用注释占位，按实际模型修改

#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

int main() {
    // ===== 动态参数（按实际模型修改）=====
    // MODEL_FILE: 模型文件（.onnx / .caffemodel / .pb / .weights）
    std::string model_file = "model.onnx";       // TODO: 替换模型路径
    // CONFIG_FILE: 结构文件（Caffe 用 .prototxt；ONNX 可为空）
    std::string config_file = "";                // TODO: 按格式填写
    // MODEL_FRAMEWORK: 框架名（"onnx"/"caffe"/"tensorflow"/"darknet"）
    std::string framework = "onnx";              // TODO: 按模型格式修改
    // INPUT_SIZE: 输入尺寸 {W, H}
    cv::Size input_size(640, 640);               // TODO: 按模型输入修改

    cv::dnn::Net net = cv::dnn::readNet(model_file, config_file, framework);
    // net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);   // TODO: GPU 可选
    // net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);

    cv::Mat img = cv::imread("example.png");     // TODO: 替换输入图片
    cv::Mat blob = cv::dnn::blobFromImage(img, 1.0 / 255.0, input_size, cv::Scalar(), true);

    net.setInput(blob);
    // 预热
    for (int i = 0; i < 3; i++) net.forward();

    auto start = std::chrono::steady_clock::now();
    cv::Mat output = net.forward();              // TODO: 指定输出层名
    auto end = std::chrono::steady_clock::now();
    std::cout << "inference time: "
              << std::chrono::duration<double, std::milli>(end - start).count()
              << " ms" << std::endl;
    std::cout << "output size: " << output.size << std::endl;  // TODO: 后处理

    return 0;
}
