
cmake_minimum_required(VERSION 3.2)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(gpu-stream)

set(gpu-stream_VERSION_MAJOR 2)
set(gpu-stream_VERSION_MINOR 0)

configure_file(common.h.in common.h)

find_package(CUDA 7.0 QUIET)
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

# TODO: Find OpenACC implementations somehow
if (true)
  add_executable(gpu-stream-acc main.cpp ACCStream.cpp)
  target_compile_definitions(gpu-stream-acc PUBLIC ACC)
  target_compile_options(gpu-stream-acc PUBLIC "-hstd=c++11")
else ()
  message("Skipping OpenACC...")
endif ()

# TODO: Find SYCL implementations somehow
if (true)
  add_executable(gpu-stream-sycl main.cpp SYCLStream.cpp)
  target_compile_definitions(gpu-stream-sycl PUBLIC SYCL)
  set_property(TARGET gpu-stream-sycl PROPERTY CXX_STANDARD 14)
else ()
  message("Skipping SYCL...")
endif ()
