cmake_minimum_required(VERSION 3.15)
project(cpp_tensorrt)

find_package(CUDA REQUIRED)
# TensorRT 头文件/库需按安装位置指定，例如：
# include_directories(/usr/include/x86_64-linux-gnu)
# link_directories(/usr/lib/x86_64-linux-gnu)

add_executable(main main.cpp)
target_link_libraries(main nvinfer nvonnxparser cudart)
