
register_flag_optional(THRUST_IMPL
        "Which Thrust implementation to use, supported options include:
         - CUDA (via https://github.com/NVIDIA/thrust)
         - ROCM (via https://github.com/ROCmSoftwarePlatform/rocThrust)
        "
        "CUDA")

register_flag_optional(SDK_DIR
        "Path to the selected Thrust implementation (e.g `/opt/nvidia/hpc_sdk/Linux_x86_64/21.9/cuda/include` for NVHPC, `/opt/rocm` for ROCm)"
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


macro(setup)
    set(CMAKE_CXX_STANDARD 14)
    if (MANAGED)
      register_definitions(MANAGED)
    endif ()

    if (${THRUST_IMPL} STREQUAL "CUDA")

        # see CUDA.cmake, we're only adding a few Thrust related libraries here

        if (POLICY CMP0104)
            cmake_policy(SET CMP0104 NEW)
        endif ()

        set(CMAKE_CUDA_ARCHITECTURES  ${CUDA_ARCH})
        # add -forward-unknown-to-host-compiler for compatibility reasons
        set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "--expt-extended-lambda " ${CUDA_EXTRA_FLAGS})
        enable_language(CUDA)
        # CMake defaults to -O2 for CUDA at Release, let's wipe that and use the global RELEASE_FLAG
        # appended later
        wipe_gcc_style_optimisation_flags(CMAKE_CUDA_FLAGS_${BUILD_TYPE})

        message(STATUS "NVCC flags: ${CMAKE_CUDA_FLAGS} ${CMAKE_CUDA_FLAGS_${BUILD_TYPE}}")


        # XXX NVHPC <= 21.9 has cub-config in `Linux_x86_64/21.9/cuda/11.4/include/cub/cmake`
        # XXX NVHPC >= 22.3 has cub-config in `Linux_x86_64/22.3/cuda/11.6/lib64/cmake/cub/`
        # same thing for thrust
        if (SDK_DIR)
            list(APPEND CMAKE_PREFIX_PATH ${SDK_DIR})
            find_package(CUB REQUIRED CONFIG PATHS ${SDK_DIR}/cub)
            find_package(Thrust REQUIRED CONFIG PATHS ${SDK_DIR}/thrust)
        else ()
            find_package(CUB REQUIRED CONFIG)
            find_package(Thrust REQUIRED CONFIG)
        endif ()

        message(STATUS "Using Thrust backend: ${BACKEND}")

        # this creates the interface that we can link to
        thrust_create_target(Thrust${BACKEND}
                HOST CPP
                DEVICE ${BACKEND})

        register_link_library(Thrust${BACKEND})
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


 
