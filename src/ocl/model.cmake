
register_flag_optional(CMAKE_CXX_COMPILER
        "Any CXX compiler that is supported by CMake detection"
        "c++")

register_flag_optional(OpenCL_LIBRARY
        "Path to OpenCL library, usually called libOpenCL.so"
        "${OpenCL_LIBRARY}")


macro(setup)
    # don't point to the CL dir as the imports already have the CL prefix
    set(OpenCL_INCLUDE_DIR "${CMAKE_SOURCE_DIR}")
    find_package(OpenCL REQUIRED)
    register_link_library(OpenCL::OpenCL)
endmacro()

