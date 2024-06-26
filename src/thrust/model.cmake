
register_flag_optional(THRUST_IMPL
        "Which Thrust implementation to use, supported options include:
         - CUDA (via https://github.com/NVIDIA/thrust or https://github.com/NVIDIA/CCCL)
         - ROCM (via https://github.com/ROCmSoftwarePlatform/rocThrust)
        "
        "CUDA")

register_flag_optional(SDK_DIR
        "Path to the installation prefix for CCCL or Thrust (e.g `/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/lib64/cmake` for NVHPC, or `/usr/local/cuda-12.5/lib64/cmake` for nvcc, or `/usr/local/cuda-11.4/include` for older nvcc, or `/opt/rocm` for ROCm)"
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
set(FETCH_CCCL_VERSION "v2.4.0" CACHE STRING "Specify version of CCCL to use if FETCH_CCCL is ON")

macro(setup)
    set(CMAKE_CXX_STANDARD 14)
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

        if (SDK_DIR)
            # CMake tries several subdirectories below SDK_DIR, see documentation:
            # https://cmake.org/cmake/help/latest/command/find_package.html#config-mode-search-procedure
            list(APPEND CMAKE_PREFIX_PATH ${SDK_DIR})
        endif ()

        set(CCCL_THRUST_DEVICE_SYSTEM ${BACKEND} CACHE STRING "" FORCE)

        # fetch CCCL if user wants to
        if (FETCH_CCCL)
            FetchContent_Declare(
                    CCCL
                    GIT_REPOSITORY https://github.com/nvidia/cccl.git
                    GIT_TAG "${FETCH_CCCL_VERSION}"
            )
            FetchContent_MakeAvailable(CCCL)
            register_link_library(CCCL::CCCL)
        else()
            # try to find CCCL locally
            find_package(CCCL CONFIG)
            if (CCCL_FOUND)
                register_link_library(CCCL::CCCL)
            else()
                # backup: find legacy projects separately
                message(WARNING "No CCCL found on your system. Trying Thrust and CUB legacy targets.")
                find_package(CUB REQUIRED CONFIG)
                find_package(Thrust REQUIRED CONFIG)
                thrust_create_target(Thrust${BACKEND} HOST CPP DEVICE ${BACKEND})
                register_link_library(Thrust${BACKEND})
            endif()
        endif()
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


 
