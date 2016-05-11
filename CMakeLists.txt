
cmake_minimum_required(VERSION 3.2)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(gpu-stream)

set(gpu-stream_VERSION_MAJOR 2)
set(gpu-stream_VERSION_MINOR 0)

configure_file(common.h.in common.h)

# If using the Cray compiler, manually add the C++11 flag because setting the
# standard through CMake as above doesn't set this flag with Cray
if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Cray")
  list(APPEND CMAKE_CXX_FLAGS -hstd=c++11)
endif ()

#-------------------------------------------------------------------------------
# CUDA
#-------------------------------------------------------------------------------
find_package(CUDA 7.0 QUIET)
set(FLAG True)
if ("${CMAKE_SYSTEM_NAME}" MATCHES "Darwin")
  execute_process(COMMAND xcodebuild -version COMMAND head -n 1 OUTPUT_VARIABLE XCODE_VERSION)
  if ("${XCODE_VERSION}" MATCHES "Xcode 7.3.1")
    message("Xcode version not supported by CUDA")
    set(FLAG False)
  endif ()
endif ()
if (${FLAG} AND ${CUDA_FOUND})
  list(APPEND CUDA_NVCC_FLAGS --std=c++11)
  cuda_add_executable(gpu-stream-cuda main.cpp CUDAStream.cu)
  target_compile_definitions(gpu-stream-cuda PUBLIC CUDA)
else ()
  message("Skipping CUDA...")
endif ()

#-------------------------------------------------------------------------------
# OpenCL
#-------------------------------------------------------------------------------
find_package(OpenCL QUIET)
if (${OpenCL_FOUND})
  add_executable(gpu-stream-ocl main.cpp OCLStream.cpp)
  target_compile_definitions(gpu-stream-ocl PUBLIC OCL)
  target_link_libraries(gpu-stream-ocl ${OpenCL_LIBRARY})
else ()
  message("Skipping OpenCL...")
endif ()

#-------------------------------------------------------------------------------
# OpenACC
#-------------------------------------------------------------------------------
# Check compiler supports an OpenACC flag
include(CheckCXXCompilerFlag)
if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")
  CHECK_CXX_COMPILER_FLAG(-fopenacc OPENACC)
  if (OPENACC)
    list (APPEND CMAKE_EXE_LINKER_FLAGS -fopenacc)
  endif ()
elseif ("${CMAKE_CXX_COMPILER_ID}" MATCHES "PGI")
  CHECK_CXX_COMPILER_FLAG(-acc OPENACC)
elseif ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Cray")
  CHECK_CXX_COMPILER_FLAG(-hacc=openacc OPENACC)
endif ()

if (OPENACC)
  add_executable(gpu-stream-acc main.cpp ACCStream.cpp)
  target_compile_definitions(gpu-stream-acc PUBLIC ACC)
else ()
  message("Skipping OpenACC...")
endif ()

#-------------------------------------------------------------------------------
# OpenMP 3.0
#-------------------------------------------------------------------------------
find_package(OpenMP QUIET)
if (${OpenMP_FOUND})
  add_executable(gpu-stream-omp3 main.cpp OMP3Stream.cpp)
  target_compile_definitions(gpu-stream-omp3 PUBLIC OMP3)
else ()
  message("Skipping OpenMP 3...")
endif ()

#-------------------------------------------------------------------------------
# OpenMP 4.5
#-------------------------------------------------------------------------------
if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Cray")
  if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 4.5)
    add_executable(gpu-stream-omp45 main.cpp OMP45Stream.cpp)
    target_compile_definitions(gpu-stream-omp45 PUBLIC OMP45)
  endif ()
endif ()

#-------------------------------------------------------------------------------
# RAJA
#-------------------------------------------------------------------------------

# TODO

#-------------------------------------------------------------------------------
# Kokkos
#-------------------------------------------------------------------------------

# TODO

#-------------------------------------------------------------------------------
# SYCL
#-------------------------------------------------------------------------------
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
