
cmake_minimum_required(VERSION 3.2)

if(NOT DEFINED HIP_PATH)
    if(NOT DEFINED ENV{HIP_PATH})
        set(HIP_PATH "/opt/rocm/hip" CACHE PATH "Path to which HIP has been installed")
    else()
        set(HIP_PATH $ENV{HIP_PATH} CACHE PATH "Path to which HIP has been installed")
    endif()
endif()
set(CMAKE_MODULE_PATH "${HIP_PATH}/cmake" ${CMAKE_MODULE_PATH})


set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(gpu-stream)

include(CheckIncludeFileCXX)
include(CheckCXXCompilerFlag)

set(gpu-stream_VERSION_MAJOR 2)
set(gpu-stream_VERSION_MINOR 1)

configure_file(common.h.in common.h)
include_directories(${CMAKE_BINARY_DIR})

# Use 'Release' if no build type specified
if (NOT CMAKE_BUILD_TYPE)
   message("No CMAKE_BUILD_TYPE specified, defaulting to 'Release'")
   set(CMAKE_BUILD_TYPE "Release")
endif()

# If using the Cray compiler, manually add the C++11 flag because setting the
# standard through CMake as above doesn't set this flag with Cray
if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Cray")
  list(APPEND CMAKE_CXX_FLAGS -hstd=c++11)
endif ()

#-------------------------------------------------------------------------------
# HIP
#-------------------------------------------------------------------------------
find_package(HIP QUIET)
if(${HIP_FOUND})
  list(APPEND HIP_HIPCC_FLAGS --std=c++11)
  hip_add_executable(gpu-stream-hip main.cpp HIPStream.cu)
  target_compile_definitions(gpu-stream-hip PUBLIC HIP)
else()
  message("Skipping HIP...")
endif()

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

if (RAJA_PATH)
  find_package(OpenMP)
  find_package(CUDA 7.5)
  list(APPEND CUDA_NVCC_FLAGS "-arch compute_35")
  list(APPEND CUDA_NVCC_FLAGS --expt-extended-lambda)
  list(APPEND CUDA_NVCC_FLAGS -Xcompiler ${OpenMP_CXX_FLAGS})
  list(APPEND CUDA_NVCC_FLAGS -DUSE_RAJA)
  cuda_include_directories(${RAJA_PATH}/include)
  set_source_files_properties(RAJAStream.cpp PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)
  cuda_add_executable(gpu-stream-raja main.cpp RAJAStream.cpp)
  target_compile_definitions(gpu-stream-raja PUBLIC USE_RAJA)
  target_link_libraries(gpu-stream-raja "-L${RAJA_PATH}/lib -lRAJA")
else()
  message("Skipping RAJA... (use -DRAJA_PATH=/path/to/raja to opt in)")
endif()

#-------------------------------------------------------------------------------
# Kokkos
#-------------------------------------------------------------------------------
if (KOKKOS_PATH)
  if ("${CMAKE_SYSTEM_NAME}" MATCHES "Linux")
    add_custom_target(gpu-stream-kokkos COMMAND make -f KokkosMakefile KOKKOS_PATH=${KOKKOS_PATH})
  else()
    message("Skipping Kokkos (requires Linux)")
  endif()
else()
  message("Skipping Kokkos... (use -DKOKKOS_PATH=/path/to/kokkos to opt in)")
endif()

#-------------------------------------------------------------------------------
# SYCL
#-------------------------------------------------------------------------------
if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang" OR
    "${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")
  # Use C++14 if available, otherwise drop back to C++11
  check_cxx_compiler_flag("-std=c++14" CXX14)
  if (CXX14)
    set(CMAKE_REQUIRED_FLAGS "-std=c++14")
  else()
    set(CMAKE_REQUIRED_FLAGS "-std=c++11")
  endif()
endif()

check_include_file_cxx("CL/sycl.hpp" HAS_SYCL)
if (HAS_SYCL)
  add_executable(gpu-stream-sycl main.cpp SYCLStream.cpp)
  target_compile_definitions(gpu-stream-sycl PUBLIC SYCL)

  # The user must define this in order to use FindComputeCpp
  if (COMPUTECPP_PACKAGE_ROOT_DIR)
    message(STATUS "Using ComputeCpp for SYCL compilation")
    include(FindComputeCpp)

    include_directories(${COMPUTECPP_INCLUDE_DIRECTORY})

    set(SOURCE_NAME "SYCLStream")

    target_compile_options(gpu-stream-sycl PUBLIC ${HOST_COMPILER_OPTIONS})
    add_sycl_to_target(gpu-stream-sycl ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_NAME}.cpp 
                      ${CMAKE_CURRENT_BINARY_DIR})
  else()
    message(STATUS "Using header-only SYCL implementation")
    set_property(TARGET gpu-stream-sycl PROPERTY CXX_STANDARD 14)
  endif()
else ()
  message("Skipping SYCL...")
endif (HAS_SYCL)
