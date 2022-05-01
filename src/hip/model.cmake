
register_flag_required(CMAKE_CXX_COMPILER
        "Absolute path to the AMD HIP C++ compiler")

register_flag_optional(USE_NONTEMPORAL_MEM
        "Flag indicating to use non-temporal memory accesses to bypass cache."
        "OFF")

# TODO: Better flag descriptions
register_flag_optional(DWORDS_PER_LANE "Flag indicating the number of double data types per wavefront lane." 4)
register_flag_optional(CHUNKS_PER_BLOCK "Flag indicating the chunks per block." 1)

macro(setup)
    # Ensure we set the proper preprocessor directives
    if (USE_NONTEMPORAL_MEM)
        add_definitions(-DNONTEMPORAL)
    endif ()
    register_definitions(DWORDS_PER_LANE=${DWORDS_PER_LANE})
    register_definitions(CHUNKS_PER_BLOCK=${CHUNKS_PER_BLOCK})
endmacro()