
cmake_minimum_required(VERSION 2.8)

project(gpu-stream)

set(CMAKE_SOURCE_DIR src/)

list(APPEND CMAKE_CXX_FLAGS --std=c++11)

find_package(CUDA REQUIRED)

cuda_add_executable(cuda.exe src/main.cpp src/CUDAStream.cu)
target_compile_definitions(cuda.exe PUBLIC CUDA)

find_package(OpenCL)
add_executable(ocl.exe src/main.cpp src/OCLStream.cpp)
target_compile_definitions(ocl.exe PUBLIC OCL)
target_link_libraries(ocl.exe ${OpenCL_LIBRARY})

