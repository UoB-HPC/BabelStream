
cmake_minimum_required(VERSION 2.8)

project(gpu-stream)

set(gpu-stream_VERSION_MAJOR 2)
set(gpu-stream_VERSION_MINOR 0)

list(APPEND CMAKE_CXX_FLAGS --std=c++11)

configure_file(src/common.h.in src/common.h)

find_package(CUDA QUIET)
if (${CUDA_FOUND})
  cuda_add_executable(cuda.exe src/main.cpp src/CUDAStream.cu)
  target_compile_definitions(cuda.exe PUBLIC CUDA)
else (${CUDA_FOUND})
  message("Skipping CUDA...")
endif (${CUDA_FOUND})

find_package(OpenCL QUIET)
if (${OpenCL_FOUND})
  add_executable(ocl.exe src/main.cpp src/OCLStream.cpp)
  target_compile_definitions(ocl.exe PUBLIC OCL)
  target_link_libraries(ocl.exe ${OpenCL_LIBRARY})
else (${OpenCL_FOUND})
  message("Skipping OpenCL...")
endif (${OpenCL_FOUND})

