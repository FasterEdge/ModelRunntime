cmake_minimum_required(VERSION 3.15)
project(cpp_libtorch)

# 需先下载 LibTorch 并设置 CMAKE_PREFIX_PATH 指向解压目录
find_package(Torch REQUIRED)

add_executable(main main.cpp)
target_link_libraries(main "${TORCH_LIBRARIES}")
set_property(TARGET main PROPERTY CXX_STANDARD 17)
