
register_flag_optional(CMAKE_CXX_COMPILER
        "Any CXX compiler that is supported by CMake detection, this is used for host compilation when required by the SYCL compiler"
        "c++")

register_flag_required(SYCL_COMPILER
        "Compile using the specified SYCL compiler implementation
        Supported values are
           ONEAPI-ICPX  - icpx as a standalone compiler
           ONEAPI-Clang - oneAPI's Clang driver (enabled via `source /opt/intel/oneapi/setvars.sh  --include-intel-llvm`)
           DPCPP        - dpc++ as a standalone compiler (https://github.com/intel/llvm)
           HIPSYCL      - hipSYCL compiler (https://github.com/illuhad/hipSYCL)
           AdaptiveCpp  - AdaptiveCpp compiler (https://github.com/adaptivecpp/adaptivecpp)")

register_flag_optional(SYCL_COMPILER_DIR
        "Absolute path to the selected SYCL compiler directory, most are packaged differently so set the path according to `SYCL_COMPILER`:
           ONEAPI-ICPX   - `icpx` must be used for OneAPI 2023 and later on releases (i.e `source /opt/intel/oneapi/setvars.sh` first)
           ONEAPI-Clang  - set to the directory that contains the Intel clang++ binary.
           AdaptiveCpp|HIPSYCL|DPCPP - set to the root of the binary distribution that contains at least `bin/`, `include/`, and `lib/`"
        "")

macro(setup)
    set(CMAKE_CXX_STANDARD 17)

    if (${SYCL_COMPILER} STREQUAL "AdaptiveCpp")
        set(adaptivecpp_DIR ${SYCL_COMPILER_DIR}/lib/cmake/adaptivecpp)

        if (NOT EXISTS "${AdaptiveCpp_DIR}")
            message(WARNING "Falling back to AdaptiveCpp < 0.9.0 CMake structure")
            set(AdaptiveCpp_DIR ${SYCL_COMPILER_DIR}/lib/cmake)
        endif ()
        if (NOT EXISTS "${AdaptiveCpp_DIR}")
            message(FATAL_ERROR "Can't find the appropriate CMake definitions for AdaptiveCpp")
        endif ()

        # register_definitions(_GLIBCXX_USE_CXX11_ABI=0)
        find_package(AdaptiveCpp CONFIG REQUIRED)
        message(STATUS "ok")
    elseif (${SYCL_COMPILER} STREQUAL "HIPSYCL")
        set(hipSYCL_DIR ${SYCL_COMPILER_DIR}/lib/cmake/hipSYCL)

        if (NOT EXISTS "${hipSYCL_DIR}")
            message(WARNING "Falling back to hipSYCL < 0.9.0 CMake structure")
            set(hipSYCL_DIR ${SYCL_COMPILER_DIR}/lib/cmake)
        endif ()
        if (NOT EXISTS "${hipSYCL_DIR}")
            message(FATAL_ERROR "Can't find the appropriate CMake definitions for hipSYCL")
        endif ()

        # register_definitions(_GLIBCXX_USE_CXX11_ABI=0)
        find_package(hipSYCL CONFIG REQUIRED)
        message(STATUS "ok")
    elseif (${SYCL_COMPILER} STREQUAL "DPCPP")
        set(CMAKE_CXX_COMPILER ${SYCL_COMPILER_DIR}/bin/clang++)
        include_directories(${SYCL_COMPILER_DIR}/include/sycl)
        register_append_cxx_flags(ANY -fsycl)
        register_append_link_flags(-fsycl)
    elseif (${SYCL_COMPILER} STREQUAL "ONEAPI-ICPX")
        set(CMAKE_CXX_COMPILER icpx)
        set(CMAKE_C_COMPILER icx)
        register_append_cxx_flags(ANY -fsycl)
        register_append_link_flags(-fsycl)
    elseif (${SYCL_COMPILER} STREQUAL "ONEAPI-Clang")
        set(CMAKE_CXX_COMPILER clang++)
        set(CMAKE_C_COMPILER clang)
        register_append_cxx_flags(ANY -fsycl)
        register_append_link_flags(-fsycl)
    else ()
        message(FATAL_ERROR "SYCL_COMPILER=${SYCL_COMPILER} is unsupported")
    endif ()

endmacro()


macro(setup_target NAME)
    if (${SYCL_COMPILER} STREQUAL "AdaptiveCpp")
        # so AdaptiveCpp has this weird (and bad) CMake usage where they append their
        # own custom integration header flags AFTER the target has been specified
        # hence this macro here
        add_sycl_to_target(
                TARGET ${NAME}
                SOURCES ${IMPL_SOURCES})
    elseif (${SYCL_COMPILER} STREQUAL "HIPSYCL")
        # so hipSYCL has this weird (and bad) CMake usage where they append their
        # own custom integration header flags AFTER the target has been specified
        # hence this macro here
        add_sycl_to_target(
                TARGET ${NAME}
                SOURCES ${IMPL_SOURCES})
    endif ()
endmacro()
