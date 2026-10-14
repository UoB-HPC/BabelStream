
register_flag_optional(CMAKE_CXX_COMPILER
        "Any CXX compiler that is supported by CMake detection, this is used for host compilation"
        "c++")

register_flag_optional(MEM "Device memory mode:
        DEFAULT   - allocate host and device memory pointers.
        MANAGED   - use CUDA Managed Memory.
        PAGEFAULT - shared memory, only host pointers allocated."
        "DEFAULT")

register_flag_optional(STRIDE "Kernel stride: GRID_STRIDE or BLOCK_STRIDE" "GRID_STRIDE")

register_flag_optional(UNROLL_FACTOR "Kernel unroll factor:" "4")


register_flag_required(CMAKE_CUDA_COMPILER
        "Path to the CUDA nvcc compiler")

# XXX we may want to drop this eventually and use CMAKE_CUDA_ARCHITECTURES directly
register_flag_required(CUDA_ARCH
        "Nvidia architecture, will be passed in via `-arch=` (e.g `sm_70`) for nvcc")

register_flag_optional(CUDA_EXTRA_FLAGS
        "Additional CUDA flags passed to nvcc, this is appended after `CUDA_ARCH`"
        "")


macro(setup)

    # XXX CMake 3.18 supports CMAKE_CUDA_ARCHITECTURES/CUDA_ARCHITECTURES but we support older CMakes
    if(POLICY CMP0104)
        cmake_policy(SET CMP0104 OLD)
    endif()

    enable_language(CUDA)
    register_definitions(${MEM})
    register_definitions(${STRIDE})

    # add -forward-unknown-to-host-compiler for compatibility reasons
    # add --extended-lambda for device-lambdas
    set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-forward-unknown-to-host-compiler" "-arch=${CUDA_ARCH}"
      "--extended-lambda" "-DUNROLL_FACTOR=${UNROLL_FACTOR}" ${CUDA_EXTRA_FLAGS})
    string(REPLACE ";" " " CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")

    # Link against the NVIDIA Management Library for device information
    register_link_library("nvidia-ml")
    
    # CMake defaults to -O2 for CUDA at Release, let's wipe that and use the global RELEASE_FLAG
    # appended later
    wipe_gcc_style_optimisation_flags(CMAKE_CUDA_FLAGS_${BUILD_TYPE})

    message(STATUS "NVCC flags: ${CMAKE_CUDA_FLAGS} ${CMAKE_CUDA_FLAGS_${BUILD_TYPE}}")
endmacro()

