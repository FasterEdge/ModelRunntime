cmake_minimum_required(VERSION 3.15)
project(cpp_tflite)

# 需先构建 TFLite C++ 库（tensorflow/lite），或使用预编译包
add_executable(main main.cc)
# target_link_libraries(main tensorflow-lite)  # TODO: 按 TFLite 构建产物调整
