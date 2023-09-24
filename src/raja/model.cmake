register_flag_optional(CMAKE_CXX_COMPILER
        "Any CXX compiler that is supported by CMake detection and RAJA.
         See https://raja.readthedocs.io/en/main/getting_started.html#build-and-install"
        "c++")

register_flag_optional(RAJA_IN_TREE
        "Absolute path to the *source* distribution directory of RAJA.
         Make sure to use the release version of RAJA or clone RAJA recursively with submodules.
         Remember to append RAJA specific flags as well, for example:
             -DRAJA_IN_TREE=... -DENABLE_OPENMP=ON -DENABLE_CUDA=ON ...
         See https://github.com/LLNL/RAJA/blob/08cbbafd2d21589ebf341f7275c229412d0fe903/CMakeLists.txt#L44 for all available options
" "")

register_flag_optional(RAJA_IN_PACKAGE
        "Use if Raja is part of a package dependency: 
        Path to installation" "")

register_flag_optional(TARGET
        "Target offload device, implemented values are CPU, NVIDIA"
        CPU)

register_flag_optional(CUDA_TOOLKIT_ROOT_DIR
        "[TARGET==NVIDIA only] Path to the CUDA toolkit directory (e.g `/opt/cuda-11.2`) if the RAJA_ENABLE_CUDA or ENABLE_CUDA flag is specified for RAJA" "")

# XXX CMake 3.18 supports CMAKE_CUDA_ARCHITECTURES/CUDA_ARCHITECTURES but we support older CMakes
register_flag_optional(CUDA_ARCH
        "[TARGET==NVIDIA only] Nvidia architecture, will be passed in via `-arch=` (e.g `sm_70`) for nvcc"
        "")

register_flag_optional(CUDA_EXTRA_FLAGS
        "[TARGET==NVIDIA only] Additional CUDA flags passed to nvcc, this is appended after `CUDA_ARCH`"
        "")

# compiler vendor and arch specific flags
set(RAJA_FLAGS_CPU_INTEL -qopt-streaming-stores=always)

macro(setup)


    if (${TARGET} STREQUAL "CPU")
        register_definitions(RAJA_TARGET_CPU)
    else ()
        register_definitions(RAJA_TARGET_GPU)
    endif ()


    if (EXISTS "${RAJA_IN_TREE}")

        message(STATUS "Building using in-tree RAJA source at `${RAJA_IN_TREE}`")

        set(CMAKE_CXX_STANDARD 14)
        # don't build anything that isn't the RAJA library itself, by default their cmake def builds everything, whyyy?
        set(ENABLE_TESTS OFF CACHE BOOL "")
        set(ENABLE_EXAMPLES OFF CACHE BOOL "")
        set(ENABLE_REPRODUCERS OFF CACHE BOOL "")
        set(ENABLE_EXERCISES OFF CACHE BOOL "")
        set(ENABLE_DOCUMENTATION OFF CACHE BOOL "")
        set(ENABLE_BENCHMARKS OFF CACHE BOOL "")
        set(ENABLE_CUDA ${ENABLE_CUDA} CACHE BOOL "" FORCE)

        # RAJA >= v2022.03.0 switched to prefixed variables, we keep the legacy ones for backwards compatibiity
        set(RAJA_ENABLE_TESTS OFF CACHE BOOL "")
        set(RAJA_ENABLE_EXAMPLES OFF CACHE BOOL "")
        set(RAJA_ENABLE_REPRODUCERS OFF CACHE BOOL "")
        set(RAJA_ENABLE_EXERCISES OFF CACHE BOOL "")
        set(RAJA_ENABLE_DOCUMENTATION OFF CACHE BOOL "")
        set(RAJA_ENABLE_BENCHMARKS OFF CACHE BOOL "")
        set(RAJA_ENABLE_CUDA ${RAJA_ENABLE_CUDA} CACHE BOOL "" FORCE)

        if (ENABLE_CUDA OR RAJA_ENABLE_CUDA)

            # RAJA still needs ENABLE_CUDA for internal use, so if either is on, assert both.
            set(RAJA_ENABLE_CUDA ON)
            set(ENABLE_CUDA ON)

            # XXX CMake 3.18 supports CMAKE_CUDA_ARCHITECTURES/CUDA_ARCHITECTURES but we support older CMakes
            if(POLICY CMP0104)
                cmake_policy(SET CMP0104 OLD)
            endif()

            # RAJA needs all the cuda stuff setup before including!
            set(CMAKE_CUDA_COMPILER ${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc)
            set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-forward-unknown-to-host-compiler -extended-lambda -arch=${CUDA_ARCH}" ${CUDA_EXTRA_FLAGS})
            list(APPEND CMAKE_CUDA_FLAGS)

            # See https://github.com/LLNL/RAJA/pull/1302
            # And https://github.com/LLNL/RAJA/pull/1339
            set(RAJA_ENABLE_VECTORIZATION OFF CACHE BOOL "")

            message(STATUS "NVCC flags: ${CMAKE_CUDA_FLAGS}")
        endif ()

        add_subdirectory(${RAJA_IN_TREE} ${CMAKE_BINARY_DIR}/raja)
        register_link_library(RAJA)
        # RAJA's cmake screws with where the binary will end up, resetting it here:
        set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
    
    elseif (EXISTS "${RAJA_IN_PACKAGE}")
        message(STATUS "Building using packaged Raja at `${RAJA_IN_PACKAGE}`")
        find_package(RAJA REQUIRED)
        register_link_library(RAJA)
    
    else ()
        message(FATAL_ERROR "Neither `${RAJA_IN_TREE}` or `${RAJA_IN_PACKAGE}` exists")
    endif ()


    if (ENABLE_CUDA)
        # RAJA needs the codebase to be compiled with nvcc, so we tell cmake to treat sources as *.cu
        enable_language(CUDA)
        set_source_files_properties(src/raja/RAJAStream.cpp PROPERTIES LANGUAGE CUDA)
        set_source_files_properties(src/main.cpp PROPERTIES LANGUAGE CUDA)
    endif ()


    register_append_compiler_and_arch_specific_cxx_flags(
            RAJA_FLAGS_CPU
            ${CMAKE_CXX_COMPILER_ID}
            ${CMAKE_SYSTEM_PROCESSOR}
    )

endmacro()
