
register_flag_optional(CMAKE_CXX_COMPILER
        "Any CXX compiler that is supported by CMake detection"
        "c++")

register_flag_optional(NVHPC_OFFLOAD
        "Enable offloading support (via the non-standard `-stdpar`) for the new NVHPC SDK.
         The values are Nvidia architectures in ccXY format will be passed in via `-gpu=` (e.g `cc70`)

         Possible values are:
           cc35  - Compile for compute capability 3.5
           cc50  - Compile for compute capability 5.0
           cc60  - Compile for compute capability 6.0
           cc62  - Compile for compute capability 6.2
           cc70  - Compile for compute capability 7.0
           cc72  - Compile for compute capability 7.2
           cc75  - Compile for compute capability 7.5
           cc80  - Compile for compute capability 8.0
           ccall - Compile for all supported compute capabilities"
        "")

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
    set(CMAKE_CXX_STANDARD 17)
    if (NVHPC_OFFLOAD)
        set(NVHPC_FLAGS -stdpar -gpu=${NVHPC_OFFLOAD})
        # propagate flags to linker so that it links with the gpu stuff as well
        register_append_cxx_flags(ANY ${NVHPC_FLAGS})
        register_append_link_flags(${NVHPC_FLAGS})
    endif ()
        if(ONE_TBB_DIR)
        set(TBB_ROOT "${ONE_TBB_DIR}") # see https://github.com/Kitware/VTK/blob/0a31a9a3c1531ae238ac96a372fec4be42282863/CMake/FindTBB.cmake#L34
        # docs on Intel's website refers to TBB_DIR which is not correct
    endif()
    if (NOT USE_TBB)
        # Only find TBB when we're not building in-tree
        find_package(TBB REQUIRED)
    endif()
        register_link_library(TBB::tbb)
    if (USE_ONEDPL)
        register_definitions(USE_ONEDPL)
        register_link_library(oneDPL)
    endif ()
endmacro()