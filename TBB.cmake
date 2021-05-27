
register_flag_required(TBB_DIR
        "Absolute path to oneTBB distribution, the directory should contains at least `include/` and `lib/")

macro(setup)
    set(CMAKE_CXX_STANDARD 14) # we use auto in lambda parameters for the different partitioners
    # see https://github.com/oneapi-src/oneTBB/blob/master/cmake/README.md#tbbconfig---integration-of-binary-packages
    find_package(TBB REQUIRED)
    register_link_library(TBB::tbb)
endmacro()
