cmake_minimum_required(VERSION 3.15)
project(cpp_onnxruntime)

find_package(onnxruntime REQUIRED)   # 或使用 -I/usr/local/include 手动指定

add_executable(main main.cpp)
target_link_libraries(main onnxruntime)
