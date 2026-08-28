cmake_minimum_required(VERSION 3.15)
project(cpp_opencv_dnn)

find_package(OpenCV REQUIRED)

add_executable(main main.cpp)
target_link_libraries(main ${OpenCV_LIBS})
