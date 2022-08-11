
register_flag_required(CMAKE_CXX_COMPILER
        "Absolute path to the AMD HIP C++ compiler")

register_flag_optional(DWORDS_PER_LANE "Flag indicating the number of dwords to process per wavefront lane." 4)

macro(setup)
    register_definitions(DWORDS_PER_LANE=${DWORDS_PER_LANE})
endmacro()