cmake_minimum_required(VERSION 3.15)
project(cpp_openvino)

find_package(OpenVINO REQUIRED)

add_executable(main main.cpp)
target_link_libraries(main openvino::runtime)
