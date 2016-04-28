
cmake_minimum_required(VERSION 2.8)

project(gpu-stream)

set(gpu-stream_VERSION_MAJOR 2)
set(gpu-stream_VERSION_MINOR 0)

list(APPEND CMAKE_CXX_FLAGS --std=c++11)

find_package(CUDA QUIET)
if (${CUDA_FOUND})
set(IMPLEMENTATION CUDA)
configure_file(src/common.h.in src/common_cuda.h)
cuda_add_executable(cuda.exe src/main.cpp src/CUDAStream.cu)
target_compile_definitions(cuda.exe PUBLIC CUDA)
else (${CUDA_FOUND})
message("Skipping CUDA...")
endif (${CUDA_FOUND})

find_package(OpenCL QUIET)
if (${OpenCL_FOUND})
set(gpu-stream_IMPLEMENTATION OpenCL)
configure_file(src/common.h.in src/common_ocl.h)
add_executable(ocl.exe src/main.cpp src/OCLStream.cpp)
target_compile_definitions(ocl.exe PUBLIC OCL)
target_link_libraries(ocl.exe ${OpenCL_LIBRARY})
else (${OpenCL_FOUND})
message("Skipping OpenCL...")
endif (${OpenCL_FOUND})

