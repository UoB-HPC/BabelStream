
register_flag_optional(CMAKE_CXX_COMPILER
        "Any CXX compiler that is supported by CMake detection"
        "c++")

register_flag_optional(OpenCL_LIBRARY
        "Path to OpenCL library, usually called libOpenCL.so"
        "${OpenCL_LIBRARY}")


macro(setup)
    setup_opencl_header_includes()
    # an empty OpenCL_LIBRARY from the flag default would stop FindOpenCL from searching
    if (NOT OpenCL_LIBRARY)
        unset(OpenCL_LIBRARY)
    endif ()
    find_package(OpenCL REQUIRED)
    register_link_library(OpenCL::OpenCL)
endmacro()

