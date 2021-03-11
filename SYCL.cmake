
register_flag_optional(CMAKE_CXX_COMPILER
        "Any CXX compiler that is supported by CMake detection, this is used for host compilation when required by the SYCL compiler"
        "c++")

register_flag_required(SYCL_COMPILER
        "Compile using the specified SYCL compiler implementation
        Supported values are
           ONEAPI-DPCPP - dpc++ that is part of an oneAPI Base Toolkit distribution (https://software.intel.com/content/www/us/en/develop/tools/oneapi/base-toolkit.html)
           DPCPP        - dpc++ as a standalone compiler (https://github.com/intel/llvm)
           HIPSYCL      - hipSYCL compiler (https://github.com/illuhad/hipSYCL)
           COMPUTECPP   - ComputeCpp compiler (https://developer.codeplay.com/products/computecpp/ce/home)")

register_flag_optional(SYCL_COMPILER_DIR
        "Absolute path to the selected SYCL compiler directory, most are packaged differently so set the path according to `SYCL_COMPILER`:
           ONEAPI-DPCPP             - not required but `dpcpp` must be on PATH, load oneAPI as per documentation (i.e `source /opt/intel/oneapi/setvars.sh` first)
           HIPSYCL|DPCPP|COMPUTECPP - set to the root of the binary distribution that contains at least `bin/`, `include/`, and `lib/`"
        "")

register_flag_optional(OpenCL_LIBRARY
        "[ComputeCpp only] Path to OpenCL library, usually called libOpenCL.so"
        "${OpenCL_LIBRARY}")

macro(setup)
    set(CMAKE_CXX_STANDARD 17)


    if (${SYCL_COMPILER} STREQUAL "HIPSYCL")


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

    elseif (${SYCL_COMPILER} STREQUAL "COMPUTECPP")

        list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules)
        set(ComputeCpp_DIR ${SYCL_COMPILER_DIR})

        # don't point to the CL dir as the imports already have the CL prefix
        set(OpenCL_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/CL")

        register_definitions(CL_TARGET_OPENCL_VERSION=220 _GLIBCXX_USE_CXX11_ABI=0)
        # ComputeCpp needs OpenCL
        find_package(ComputeCpp REQUIRED)

        # this must come after FindComputeCpp (!)
        set(COMPUTECPP_USER_FLAGS -O3 -no-serial-memop)

    elseif (${SYCL_COMPILER} STREQUAL "DPCPP")
        set(CMAKE_CXX_COMPILER ${SYCL_COMPILER_DIR}/bin/clang++)
        include_directories(${SYCL_COMPILER_DIR}/include/sycl)
        register_definitions(CL_TARGET_OPENCL_VERSION=220)
        register_append_cxx_flags(ANY -fsycl)
        register_append_link_flags(-fsycl)
    elseif (${SYCL_COMPILER} STREQUAL "ONEAPI-DPCPP")
        set(CMAKE_CXX_COMPILER dpcpp)
        register_definitions(CL_TARGET_OPENCL_VERSION=220)
    else ()
        message(FATAL_ERROR "SYCL_COMPILER=${SYCL_COMPILER} is unsupported")
    endif ()

endmacro()


macro(setup_target NAME)
    if (
    (${SYCL_COMPILER} STREQUAL "COMPUTECPP") OR
    (${SYCL_COMPILER} STREQUAL "HIPSYCL"))
        # so ComputeCpp and hipSYCL has this weird (and bad) CMake usage where they append their
        # own custom integration header flags AFTER the target has been specified
        # hence this macro here
        add_sycl_to_target(
                TARGET ${NAME}
                SOURCES ${IMPL_SOURCES})
    endif ()
endmacro()
