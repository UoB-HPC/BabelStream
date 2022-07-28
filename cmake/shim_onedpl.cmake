

if (USE_ONEDPL)
    #        # XXX see https://www.intel.com/content/www/us/en/develop/documentation/oneapi-dpcpp-library-guide/top/oneapi-dpc-library-onedpl-overview.html
    #        # this is to avoid the system TBB headers (if exists) from having precedence which isn't compatible with oneDPL's par implementation
    #        register_definitions(
    #                PSTL_USE_PARALLEL_POLICIES=0
    #                _GLIBCXX_USE_TBB_PAR_BACKEND=0
    #        )
    register_definitions(USE_ONEDPL)
    if (USE_ONEDPL STREQUAL "TBB")
        register_definitions(ONEDPL_USE_TBB_BACKEND=1)
        # TBB will either be linked later (USE_TBB==ON) or via extra libraries, don't do anything here
    elseif (USE_ONEDPL STREQUAL "OPENMP")
        register_definitions(ONEDPL_USE_OPENMP_BACKEND=1)
        # Link OpenMP via CMAKE
        find_package(OpenMP REQUIRED)
        register_link_library(OpenMP::OpenMP_CXX)
    elseif (USE_ONEDPL STREQUAL "SYCL")
        register_definitions(ONEDPL_USE_DPCPP_BACKEND=1)
        # This needs a SYCL compiler, will fail if CXX doesn't SYCL2020
        register_append_cxx_flags(ANY -fsycl-unnamed-lambda -fsycl)
    else ()
        message(FATAL_ERROR "Unsupported USE_ONEDPL backend: ${USE_ONEDPL}, see USE_ONEDPL flag description for available values.")
    endif ()
    register_directories(ANY ${onedpl_SOURCE_DIR}/include)
endif ()