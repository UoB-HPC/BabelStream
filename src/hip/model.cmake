
register_flag_required(CMAKE_CXX_COMPILER
        "Absolute path to the AMD HIP C++ compiler")

register_flag_optional(MEM "Device memory mode:
        DEFAULT   - allocate host and device memory pointers.
        MANAGED   - use HIP Managed Memory.
        PAGEFAULT - shared memory, only host pointers allocated."
        "DEFAULT")

macro(setup)
    # nothing to do here as hipcc does everything correctly, what a surprise!
    register_definitions(${MEM})
endmacro()