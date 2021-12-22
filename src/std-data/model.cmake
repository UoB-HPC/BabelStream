
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

register_flag_optional(ONEDPL_OFFLOAD
        "Use the DPC++ oneDPL library which supports STL algorithms on SYCL, TBB, and OpenMP.
         This option only supports the oneDPL library shipped with oneAPI, and must use the dpcpp
         compiler (i.e -DCMAKE_CXX_COMPILER=dpcpp) for this option.
         Make sure your oneAPI installation includes at least the following components: dpcpp, onedpl, onetbb.
         The env. variable `TBBROOT` needs to point to the base directory of your TBB install (e.g /opt/intel/oneapi/tbb/latest/).
         This should be done by oneAPI's `setvars.sh` script automatically.

         Possible values are:
           TBB   - Use the TBB backend, the correct TBB library will be linked from oneAPI automatically.
           OMP   - Use the OpenMP backend
           DPCPP - Use the SYCL (via dpcpp) backend with the default selector.
                  See https://intel.github.io/llvm-docs/EnvironmentVariables.html#sycl-device-filter
                  on selecting a non-default device or SYCL backend."
        "")


macro(setup)
    set(CMAKE_CXX_STANDARD 17)

    if (NVHPC_OFFLOAD AND ONEDPL_OFFLOAD)
        message(FATAL_ERROR "NVHPC_OFFLOAD and NVHPC_OFFLOAD are mutually exclusive")
    endif ()

    if (NVHPC_OFFLOAD)
        set(NVHPC_FLAGS -stdpar -gpu=${NVHPC_OFFLOAD})
        # propagate flags to linker so that it links with the gpu stuff as well
        register_append_cxx_flags(ANY ${NVHPC_FLAGS})
        register_append_link_flags(${NVHPC_FLAGS})
    endif ()


    if (ONEDPL_OFFLOAD)
        set(CXX_EXTRA_FLAGS)
        set(CXX_EXTRA_LIBRARIES /opt/intel/oneapi/tbb/2021.4.0/lib/intel64/gcc4.8/libtbb.so)
        # propagate flags to linker so that it links with the gpu stuff as well
        register_append_cxx_flags(ANY -fopenmp -fsycl-unnamed-lambda -fsycl)

        # XXX see https://www.intel.com/content/www/us/en/develop/documentation/oneapi-dpcpp-library-guide/top/oneapi-dpc-library-onedpl-overview.html
        # this is to avoid the system TBB headers (if exists) from having precedence which isn't compatible with oneDPL's par implementation
        register_definitions(
                PSTL_USE_PARALLEL_POLICIES=0
                _GLIBCXX_USE_TBB_PAR_BACKEND=0
        )

        register_definitions(USE_ONEDPL)
        if (ONEDPL_OFFLOAD STREQUAL "TBB")
            register_definitions(ONEDPL_USE_TBB_BACKEND=1)
        elseif (ONEDPL_OFFLOAD STREQUAL "OPENMP")
            register_definitions(ONEDPL_USE_OPENMP_BACKEND=1)
        elseif (ONEDPL_OFFLOAD STREQUAL "SYCL")
            register_definitions(ONEDPL_USE_DPCPP_BACKEND=1)
        else ()
            message(FATAL_ERROR "Unsupported ONEDPL_OFFLOAD backend: ${ONEDPL_OFFLOAD}")
        endif ()

        # even with the workaround above, -ltbb may still end up with the wrong one, so be explicit here
        register_link_library($ENV{TBBROOT}/lib/intel64/gcc4.8/libtbb.so)

    endif ()

endmacro()
