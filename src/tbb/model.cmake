
register_flag_optional(ONE_TBB_DIR
        "Absolute path to oneTBB (with header `onetbb/tbb.h`) distribution, the directory should contain at least `include/` and `lib/.
         If unspecified, the system TBB (with header `tbb/tbb.h`) will be used via CMake's find_package(TBB)."
        "")


register_flag_optional(PARTITIONER
        "Partitioner specifies how a loop template should partition its work among threads.
         Possible values are:
            AUTO     - Optimize range subdivision based on work-stealing events.
            AFFINITY - Proportional splitting that optimizes for cache affinity.
            STATIC   - Distribute work uniformly with no additional load balancing.
            SIMPLE   - Recursively split its range until it cannot be further subdivided.
            See https://spec.oneapi.com/versions/latest/elements/oneTBB/source/algorithms.html#partitioners for more details."
        "AUTO")

register_flag_optional(USE_VECTOR
        "Whether to use std::vector<T> for storage or use aligned_alloc. C++ vectors are *zero* initialised where as aligned_alloc is uninitialised before first use."
        "OFF")

register_flag_optional(USE_TBB
        "No-op if ONE_TBB_DIR is set. Link against an in-tree oneTBB via FetchContent_Declare, see top level CMakeLists.txt for details."
        "OFF")

macro(setup)
    if(ONE_TBB_DIR)
        set(TBB_ROOT "${ONE_TBB_DIR}") # see https://github.com/Kitware/VTK/blob/0a31a9a3c1531ae238ac96a372fec4be42282863/CMake/FindTBB.cmake#L34
        # docs on Intel's website refers to TBB_DIR which is not correct
    endif()
    if (NOT USE_TBB)
        # Only find TBB when we're not building in-tree
        find_package(TBB REQUIRED)
    endif()

    # see https://github.com/oneapi-src/oneTBB/blob/master/cmake/README.md#tbbconfig---integration-of-binary-packages
    register_link_library(TBB::tbb)
    register_definitions(PARTITIONER_${PARTITIONER})
    if(USE_VECTOR)
        register_definitions(USE_VECTOR)
    endif()
endmacro()
