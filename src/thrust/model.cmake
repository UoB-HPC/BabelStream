
register_flag_optional(THRUST_IMPL
        "Which Thrust implementation to use, supported options include:
         - CUDA (via https://github.com/NVIDIA/CCCL (CUDA Core Compute Libraries))
         - ROCM (via https://github.com/ROCmSoftwarePlatform/rocThrust)
        "
        "CUDA")

register_flag_optional(SDK_DIR
        "Path to the installation prefix for CCCL or Thrust (e.g `/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/lib64/cmake` for NVHPC, or `/usr/local/cuda-13.0/lib64/cmake` for nvcc, or `/opt/rocm` for ROCm)"
        "")

register_flag_optional(BACKEND
        "[THRUST_IMPL==CUDA] CUDA's Thrust implementation supports the following backends:
          - CUDA
          - OMP
          - TBB
        "
        "CUDA")

register_flag_optional(MANAGED "Enabled managed memory mode."
        "OFF")

register_flag_optional(CMAKE_CUDA_COMPILER
        "[THRUST_IMPL==CUDA] Path to the CUDA nvcc compiler"
        "")

# XXX we may want to drop this eventually and use CMAKE_CUDA_ARCHITECTURES directly
register_flag_optional(CUDA_ARCH
        "[THRUST_IMPL==CUDA] Nvidia architecture, will be passed in via `-arch=` (e.g `sm_70`) for nvcc"
        "")

register_flag_optional(CUDA_EXTRA_FLAGS
        "[THRUST_IMPL==CUDA] Additional CUDA flags passed to nvcc, this is appended after `CUDA_ARCH`"
        "")

option(FETCH_CCCL "Fetch (download) the CCCL library. This uses CMake's FetchContent feature.
                   Specify version by setting FETCH_CCCL_VERSION" OFF)
set(FETCH_CCCL_VERSION "v3.1.0" CACHE STRING "Specify version of CCCL to use if FETCH_CCCL is ON")

macro(setup)
    set(CMAKE_CXX_STANDARD 17)
    if (MANAGED)
      register_definitions(MANAGED)
    endif ()

    if (${THRUST_IMPL} STREQUAL "CUDA")
        if (POLICY CMP0104)
            cmake_policy(SET CMP0104 NEW)
        endif ()
        set(CMAKE_CUDA_ARCHITECTURES  ${CUDA_ARCH})
        set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "--expt-extended-lambda " ${CUDA_EXTRA_FLAGS})
        enable_language(CUDA)
        # CMake defaults to -O2 for CUDA at Release, let's wipe that and use the global RELEASE_FLAG appended later
        wipe_gcc_style_optimisation_flags(CMAKE_CUDA_FLAGS_${BUILD_TYPE})

        message(STATUS "NVCC flags: ${CMAKE_CUDA_FLAGS} ${CMAKE_CUDA_FLAGS_${BUILD_TYPE}}")

        # append SDK_DIR to help finding CCCL
        if (SDK_DIR)
            # CMake tries several subdirectories below SDK_DIR, see documentation:
            # https://cmake.org/cmake/help/latest/command/find_package.html#config-mode-search-procedure
            list(APPEND CMAKE_PREFIX_PATH ${SDK_DIR})
        endif ()

        # append CUDA Toolkit cmake config dir to help finding CCCL
        find_package(CUDAToolkit REQUIRED)
        list(APPEND CMAKE_PREFIX_PATH "${CUDAToolkit_LIBRARY_DIR}/cmake")

        set(CCCL_THRUST_DEVICE_SYSTEM ${BACKEND} CACHE STRING "" FORCE)

        # fetch CCCL if user wants to, otherwise just try to find it
        if (FETCH_CCCL)
            FetchContent_Declare(
                    CCCL
                    GIT_REPOSITORY https://github.com/nvidia/cccl.git
                    GIT_TAG "${FETCH_CCCL_VERSION}"
            )
            FetchContent_MakeAvailable(CCCL)
        else()
            find_package(CCCL CONFIG REQUIRED)
        endif()
        register_link_library(CCCL::CCCL)
    elseif (${THRUST_IMPL} STREQUAL "ROCM")
        if (SDK_DIR)
            find_package(rocprim REQUIRED CONFIG PATHS ${SDK_DIR}/rocprim)
            find_package(rocthrust REQUIRED CONFIG PATHS ${SDK_DIR}/rocthrust)
        else ()
            find_package(rocprim REQUIRED CONFIG)
            find_package(rocthrust REQUIRED CONFIG)
        endif ()

        # for HIP we treat *.cu files as CXX otherwise CMake doesn't compile them
        set_source_files_properties(${IMPL_SOURCES} PROPERTIES LANGUAGE CXX)

        register_link_library(roc::rocthrust)
    else ()
        message(FATAL_ERROR "Unsupported THRUST_IMPL provided: ${THRUST_IMPL}")
    endif ()


endmacro()


 
