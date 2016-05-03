
cmake_minimum_required(VERSION 2.8)

project(gpu-stream)

set(gpu-stream_VERSION_MAJOR 2)
set(gpu-stream_VERSION_MINOR 0)

list(APPEND CMAKE_CXX_FLAGS --std=c++11)

configure_file(common.h.in common.h)

find_package(CUDA QUIET)
if (${CUDA_FOUND})
  cuda_add_executable(gpu-stream-cuda main.cpp CUDAStream.cu)
  target_compile_definitions(gpu-stream-cuda PUBLIC CUDA)
else (${CUDA_FOUND})
  message("Skipping CUDA...")
endif (${CUDA_FOUND})

find_package(OpenCL QUIET)
if (${OpenCL_FOUND})
  add_executable(gpu-stream-ocl main.cpp OCLStream.cpp)
  target_compile_definitions(gpu-stream-ocl PUBLIC OCL)
  target_link_libraries(gpu-stream-ocl ${OpenCL_LIBRARY})
else (${OpenCL_FOUND})
  message("Skipping OpenCL...")
endif (${OpenCL_FOUND})

