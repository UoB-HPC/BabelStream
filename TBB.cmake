
register_flag_optional(ONE_TBB_DIR
        "Absolute path to oneTBB (with header `onetbb/tbb.h`) distribution, the directory should contain at least `include/` and `lib/.
         If unspecified, the system TBB (with header `tbb/tbb.h`) will be used via CMake's find_package(TBB)." 
        "")

macro(setup)
    if(ONE_TBB_DIR)
        set(TBB_ROOT "${ONE_TBB_DIR}") # see https://github.com/Kitware/VTK/blob/0a31a9a3c1531ae238ac96a372fec4be42282863/CMake/FindTBB.cmake#L34
        # docs on Intel's website refers to TBB_DIR which hasn't been correct for 6 years
    endif()
    

    set(CMAKE_CXX_STANDARD 14) # we use auto in lambda parameters for the different partitioners
    # see https://github.com/oneapi-src/oneTBB/blob/master/cmake/README.md#tbbconfig---integration-of-binary-packages
    find_package(TBB REQUIRED)
    register_link_library(TBB::tbb)
endmacro()
