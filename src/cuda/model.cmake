
register_flag_optional(CMAKE_CXX_COMPILER
        "Any CXX compiler that is supported by CMake detection, this is used for host compilation"
        "c++")

register_flag_optional(MEM "Device memory mode:
        DEFAULT   - allocate host and device memory pointers.
        MANAGED   - use CUDA Managed Memory.
        PAGEFAULT - shared memory, only host pointers allocated."
        "DEFAULT")

register_flag_required(CMAKE_CUDA_COMPILER
        "Path to the CUDA nvcc compiler")

# XXX we may want to drop this eventually and use CMAKE_CUDA_ARCHITECTURES directly
register_flag_required(CUDA_ARCH
        "Nvidia architecture, will be passed in via `-arch=` (e.g `sm_70`) for nvcc")

register_flag_optional(CUDA_CLANG_DRIVER
        "Disable any nvcc-specific flags so that setting CMAKE_CUDA_COMPILER to clang++ can compile successfully"
        "OFF")

register_flag_optional(CUDA_EXTRA_FLAGS
        "Additional CUDA flags passed to the CUDA compiler, this is appended after `CUDA_ARCH`"
        "")


macro(setup)

    # XXX CMake 3.18 supports CMAKE_CUDA_ARCHITECTURES/CUDA_ARCHITECTURES but we support older CMakes
    if (POLICY CMP0104)
        cmake_policy(SET CMP0104 OLD)
    endif ()

    register_definitions(${MEM})

    if (CUDA_CLANG_DRIVER)
        if (CMAKE_VERSION VERSION_LESS "3.18.0")
            message(FATAL_ERROR "Using clang driver for CUDA is only supported for CMake >= 3.18")
        endif ()
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++17 --cuda-gpu-arch=${CUDA_ARCH} ${CUDA_EXTRA_FLAGS}")
    else ()
        # add -forward-unknown-to-host-compiler for compatibility reasons
        # add -std=c++17 manually as older CMake seems to omit this (source gets treated as C otherwise)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++17 -forward-unknown-to-host-compiler -arch=${CUDA_ARCH} ${CUDA_EXTRA_FLAGS}")
    endif ()
    string(REPLACE ";" " " CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")

    enable_language(CUDA)

    # CMake defaults to -O2 for CUDA at Release, let's wipe that and use the global RELEASE_FLAG
    # appended later
    wipe_gcc_style_optimisation_flags(CMAKE_CUDA_FLAGS_${BUILD_TYPE})

    message(STATUS "CUDA compiler flags: ${CMAKE_CUDA_FLAGS} ${CMAKE_CUDA_FLAGS_${BUILD_TYPE}}")
endmacro()

