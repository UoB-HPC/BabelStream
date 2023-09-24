
register_flag_optional(CMAKE_CXX_COMPILER
        "Any CXX compiler that is supported by CMake detection and supports C++20 Ranges"
        "c++")

register_flag_optional(USE_TBB
        "No-op if ONE_TBB_DIR is set. Link against an in-tree oneTBB via FetchContent_Declare, see top level CMakeLists.txt for details."
        "OFF")

register_flag_optional(USE_ONEDPL
        "Link oneDPL which implements C++17 executor policies (via execution_policy_tag) for different backends.

        Possible values are:
          OPENMP - Implements policies using OpenMP.
                   CMake will handle any flags needed to enable OpenMP if the compiler supports it.
          TBB    - Implements policies using TBB.
                   TBB must be linked via USE_TBB or be available in LD_LIBRARY_PATH.
          DPCPP  - Implements policies through SYCL2020.
                   This requires the DPC++ compiler (other SYCL compilers are untested), required SYCL flags are added automatically."
        "OFF")

macro(setup)

    # TODO this needs to eventually be removed when CMake adds proper C++20 support or at least update the flag used here

    # C++ 2a is too new, disable CMake's std flags completely:
    set(CMAKE_CXX_EXTENSIONS OFF)
    set(CMAKE_CXX_STANDARD_REQUIRED OFF)
    unset(CMAKE_CXX_STANDARD) # drop any existing standard we have set by default
    # and append our own:
    register_append_cxx_flags(ANY -std=c++20)
    if (USE_TBB)
        register_link_library(TBB::tbb)
    endif ()
    if (USE_ONEDPL)
        register_definitions(USE_ONEDPL)
        register_link_library(oneDPL)
    endif ()
endmacro()

macro(setup_target NAME)
    if (USE_ONEDPL)
        target_compile_features(${NAME} INTERFACE cxx_std_20)
        target_compile_features(oneDPL INTERFACE cxx_std_20)
    endif ()
endmacro()
