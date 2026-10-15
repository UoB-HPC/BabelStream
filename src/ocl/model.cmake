
register_flag_optional(CMAKE_CXX_COMPILER
        "Any CXX compiler that is supported by CMake detection"
        "c++")

register_flag_optional(OpenCL_LIBRARY
        "Path to OpenCL library, usually called libOpenCL.so"
        "${OpenCL_LIBRARY}")

register_flag_optional(OCL_DISABLE_MEMORY_CHECK
        "Skip the check that the buffers fit in the device's reported memory, for devices that report it incorrectly"
        "OFF")


macro(setup)
    setup_opencl_header_includes()
    # an empty OpenCL_LIBRARY from the flag default would stop FindOpenCL from searching
    if (NOT OpenCL_LIBRARY)
        unset(OpenCL_LIBRARY)
    endif ()
    find_package(OpenCL REQUIRED)
    register_link_library(OpenCL::OpenCL)
    if (OCL_DISABLE_MEMORY_CHECK)
        register_definitions(OCL_DISABLE_MEMORY_CHECK)
    endif ()
endmacro()

