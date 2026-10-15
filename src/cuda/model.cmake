
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
        "Set to ON when CMAKE_CUDA_COMPILER is clang++, this replaces the nvcc-specific flags (requires CMake >= 3.18)"
        "OFF")

register_flag_optional(ACPP_PCUDA_DRIVER
        "Set to ON to compile with AdaptiveCpp's PCUDA frontend, set CMAKE_CXX_COMPILER to acpp, CMAKE_CUDA_COMPILER and CUDA_ARCH are unused"
        "OFF")

register_flag_optional(CUDA_EXTRA_FLAGS
        "Additional CUDA flags passed to the CUDA compiler, this is appended after `CUDA_ARCH`"
        "")


macro(setup)

    # XXX CMake 3.18 supports CMAKE_CUDA_ARCHITECTURES/CUDA_ARCHITECTURES but we support older CMakes
    # CUDA_ARCH is passed to nvcc directly, so turn CMake's architecture flags off
    if(POLICY CMP0104)
        cmake_policy(SET CMP0104 NEW)
        set(CMAKE_CUDA_ARCHITECTURES OFF)
    endif()

    register_definitions(${MEM})

    if (ACPP_PCUDA_DRIVER)
        set(CMAKE_CXX_STANDARD 17)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --acpp-pcuda --acpp-pcuda-chevron-launch ${CUDA_EXTRA_FLAGS}")
        set_source_files_properties(${IMPL_SOURCES} PROPERTIES LANGUAGE CXX)
        message(STATUS "PCUDA flags: ${CMAKE_CXX_FLAGS}")
    else ()
        if (CUDA_CLANG_DRIVER)
            if (CMAKE_VERSION VERSION_LESS "3.18.0")
                message(FATAL_ERROR "Using clang as the CUDA compiler requires CMake >= 3.18")
            endif ()
            set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "--cuda-gpu-arch=${CUDA_ARCH}" ${CUDA_EXTRA_FLAGS})
        else ()
            # add -forward-unknown-to-host-compiler for compatibility reasons
            set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-forward-unknown-to-host-compiler" "-arch=${CUDA_ARCH}" ${CUDA_EXTRA_FLAGS})
        endif ()
        string(REPLACE ";" " " CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")

        # the flags must be set first as compiler detection already needs the architecture
        enable_language(CUDA)

        # CMake defaults to -O2 for CUDA at Release, let's wipe that and use the global RELEASE_FLAG
        # appended later
        wipe_gcc_style_optimisation_flags(CMAKE_CUDA_FLAGS_${BUILD_TYPE})

        message(STATUS "CUDA compiler flags: ${CMAKE_CUDA_FLAGS} ${CMAKE_CUDA_FLAGS_${BUILD_TYPE}}")
    endif ()
endmacro()

