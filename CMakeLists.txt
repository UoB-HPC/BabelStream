
cmake_minimum_required(VERSION 3.2)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(gpu-stream)

set(gpu-stream_VERSION_MAJOR 2)
set(gpu-stream_VERSION_MINOR 0)

configure_file(common.h.in common.h)

find_package(CUDA 7.0 QUIET)
set(FLAG True)
if ("${CMAKE_SYSTEM_NAME}" MATCHES "Darwin")
  execute_process(COMMAND xcodebuild -version COMMAND head -n 1 OUTPUT_VARIABLE XCODE_VERSION)
  if ("${XCODE_VERSION}" MATCHES "Xcode 7.3.1")
    message("Xcode version not supported by CUDA")
    set(FLAG False)
  endif ("${XCODE_VERSION}" MATCHES "Xcode 7.3.1")
endif ("${CMAKE_SYSTEM_NAME}" MATCHES "Darwin")
if (${FLAG} AND ${CUDA_FOUND})
  list(APPEND CUDA_NVCC_FLAGS --std=c++11)
  cuda_add_executable(gpu-stream-cuda main.cpp CUDAStream.cu)
  target_compile_definitions(gpu-stream-cuda PUBLIC CUDA)
else (${FLAG} AND ${CUDA_FOUND})
  message("Skipping CUDA...")
endif (${FLAG} AND ${CUDA_FOUND})

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
  # ComputeCpp
  # TODO: Sort this out properly!
  add_custom_target(gpu-stream-sycl
                    COMMAND compute++ ${CMAKE_CURRENT_SOURCE_DIR}/SYCLStream.cpp -sycl -no-serial-memop -O2 -emit-llvm -o SYCLStream.bc -c
                    COMMAND /usr/bin/c++ ${CMAKE_CURRENT_SOURCE_DIR}/SYCLStream.cpp -O2 -std=c++11 -include SYCLStream.sycl -o SYCLStream.o -c
                    COMMAND /usr/bin/c++ ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp -O2 -std=c++11 SYCLStream.o -include SYCLStream.sycl -lSYCL -lOpenCL -o gpu-stream-sycl -DSYCL
                    )

  # triSYCL
  #add_executable(gpu-stream-sycl main.cpp SYCLStream.cpp)
  #target_compile_definitions(gpu-stream-sycl PUBLIC SYCL)
  #set_property(TARGET gpu-stream-sycl PROPERTY CXX_STANDARD 14)
else ()
  message("Skipping SYCL...")
endif ()
