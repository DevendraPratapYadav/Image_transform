cmake_minimum_required(VERSION 2.8.12)

project(imgProc)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(imgProc ImageProcess.cpp)
target_link_libraries(imgProc ${OpenCV_LIBS})
