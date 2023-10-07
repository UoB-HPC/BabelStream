cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
    cmake_policy(SET CMP0135 NEW)
endif ()

project(BabelStream VERSION 5.0 LANGUAGES CXX C)

# uncomment for debugging build issues:
#set(CMAKE_VERBOSE_MAKEFILE ON)

# some nicer defaults for standard C++
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
include(FetchContent)

FetchContent_Declare(
        opencl_header
        URL https://github.com/KhronosGroup/OpenCL-Headers/archive/refs/tags/v2021.06.30.zip
        URL_HASH MD5=af7ab7918a6a11c60370c8651a9f0192
)

macro(setup_opencl_header_includes)
    FetchContent_GetProperties(opencl_header)
    if (NOT opencl_header_POPULATED)
        FetchContent_Populate(opencl_header)
        set(OpenCL_INCLUDE_DIR ${opencl_header_SOURCE_DIR})
    endif ()
endmacro()

# the final executable name
set(EXE_NAME babelstream)

# for chrono and some basic CXX features, models can overwrite this if required
set(CMAKE_CXX_STANDARD 11)

if (NOT CMAKE_BUILD_TYPE)
    message("No CMAKE_BUILD_TYPE specified, defaulting to 'Release'")
    set(CMAKE_BUILD_TYPE "Release")
endif ()

string(TOUPPER ${CMAKE_BUILD_TYPE} BUILD_TYPE)
if ((NOT BUILD_TYPE STREQUAL RELEASE) AND (NOT BUILD_TYPE STREQUAL DEBUG))
    message(FATAL_ERROR "Only Release or Debug is supported, got `${CMAKE_BUILD_TYPE}`")
endif ()

# setup some defaults flags for everything
set(DEFAULT_DEBUG_FLAGS -O2 -fno-omit-frame-pointer)
set(DEFAULT_RELEASE_FLAGS -O3 -march=native)

macro(hint_flag FLAG DESCRIPTION)
    if (NOT DEFINED ${FLAG})
        message(STATUS "${FLAG}: ${DESCRIPTION}")
    else ()
        # i.e. `-DFOO="-a -b"` becomes CMake's semicolon separated list `FOO=`-a;-b`
        separate_arguments(${FLAG})
    endif ()
endmacro()

# hint common extra flag options for all models if they are not set
hint_flag(CXX_EXTRA_FLAGS "
        Appends to common compile flags. These will be appended at link phase as well.
        To use separate flags at link phase, set `CXX_EXTRA_LINK_FLAGS`")
hint_flag(CXX_EXTRA_LINK_FLAGS "
        Appends to link flags which appear *before* the objects.
        Do not use this for linking libraries, as the link line is order-dependent")
hint_flag(CXX_EXTRA_LIBRARIES "
        Append to link flags which appear *after* the objects.
        Use this for linking extra libraries (e.g `-lmylib`, or simply `mylib`)")
hint_flag(CXX_EXTRA_LINKER_FLAGS "
        Append to linker flags (i.e GCC's `-Wl` or equivalent)")

# Honor user's CXX_EXTRA_LINK_FLAGS
set(CXX_EXTRA_LINK_FLAGS ${CXX_EXTRA_FLAGS} ${CXX_EXTRA_LINK_FLAGS})

option(USE_TBB "Enable the oneTBB library for *supported* models. Enabling this on models that
                don't explicitly link against TBB is a no-op, see description of your selected
                model on how this is used." OFF)

option(FETCH_TBB "Fetch (download) the oneTBB library for *supported* models. This uses CMake's
                  FetchContent feature. Specify version by setting FETCH_TBB_VERSION" OFF)
set(FETCH_TBB_VERSION "v2021.10.0" CACHE STRING "Specify version of oneTBB to use if FETCH_TBB is ON")

if (FETCH_TBB)
    FetchContent_Declare(
            TBB
            GIT_REPOSITORY https://github.com/oneapi-src/oneTBB.git
            GIT_TAG "${FETCH_TBB_VERSION}"
    )
    # Don't fail builds on waring (TBB has -Wall while not being free of warnings from unused symbols...)
    set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
    set(TBB_STRICT OFF)
    # Not using FetchContent_MakeAvailable (CMake>= 3.14) because we need EXCLUDE_FROM_ALL
    FetchContent_GetProperties(TBB)
    if (NOT TBB_POPULATED)
        FetchContent_Populate(TBB)
        add_subdirectory(${tbb_SOURCE_DIR} ${tbb_BINARY_DIR} EXCLUDE_FROM_ALL)
    endif ()
endif ()

option(USE_ONEDPL "Enable the oneDPL library for *supported* models. Enabling this on models that
                   don't explicitly link against DPL is a no-op, see description of your selected
                   model on how this is used." OFF)

option(FETCH_ONEDPL "Fetch (download) the oneDPL library for *supported* models. This uses CMake's
                  FetchContent feature. Specify version by setting FETCH_ONEDPL_VERSION" OFF)
set(FETCH_ONEDPL_VERSION "oneDPL-2022.2.0-rc1" CACHE STRING "Specify version of oneTBB to use if FETCH_ONEDPL is ON")

if (FETCH_ONEDPL)
    FetchContent_Declare(
            oneDPL
            GIT_REPOSITORY https://github.com/oneapi-src/oneDPL.git
            GIT_TAG "${FETCH_ONEDPL_VERSION}"
    )
    string(TOLOWER ${USE_ONEDPL} ONEDPL_BACKEND)
    # XXX oneDPL looks for omp instead of openmp, which mismatches(!) with ONEDPL_PAR_BACKEND if using find_package
    if (ONEDPL_BACKEND STREQUAL "openmp")
        set(ONEDPL_BACKEND omp)
    endif ()
    # Not using FetchContent_MakeAvailable (CMake>= 3.14) because we need EXCLUDE_FROM_ALL
    FetchContent_GetProperties(oneDPL)
    if (NOT oneDPL_POPULATED)
        FetchContent_Populate(oneDPL)
        if (USE_TBB)
            macro(find_package NAME)
                if ("${NAME}" STREQUAL "TBB")
                    message(STATUS "Discarding oneDPL's call to find_package(${NAME} ${ARGN})")
                else ()
                    _find_package(${NAME} ${ARGN})
                endif ()
            endmacro()
        endif ()
        add_subdirectory(${onedpl_SOURCE_DIR} ${onedpl_BINARY_DIR} EXCLUDE_FROM_ALL)

        # Fixup oneDPL's omission on setting DPCPP definitions.
        # We do this after the creation of the oneDPL target.
        if (ONEDPL_BACKEND MATCHES "^(dpcpp|dpcpp_only)$")
            target_compile_definitions(oneDPL INTERFACE ONEDPL_USE_DPCPP_BACKEND=1)
        endif ()

    endif ()
endif ()


# include our macros
include(cmake/register_models.cmake)

# register out models <model_name> <preprocessor_def_name> <source files...>
register_model(omp OMP OMPStream.cpp)
register_model(ocl OCL OCLStream.cpp)
register_model(std-data STD_DATA STDDataStream.cpp)
register_model(std-indices STD_INDICES STDIndicesStream.cpp)
register_model(std-ranges STD_RANGES STDRangesStream.cpp)
register_model(hip HIP HIPStream.cpp)
register_model(cuda CUDA CUDAStream.cu)
register_model(kokkos KOKKOS KokkosStream.cpp)
register_model(sycl SYCL SYCLStream.cpp)
register_model(sycl2020-acc SYCL2020 SYCLStream2020.cpp)
register_model(sycl2020-usm SYCL2020 SYCLStream2020.cpp)
register_model(acc ACC ACCStream.cpp)
# defining RAJA collides with the RAJA namespace so USE_RAJA
register_model(raja USE_RAJA RAJAStream.cpp)
register_model(tbb TBB TBBStream.cpp)
register_model(thrust THRUST ThrustStream.cu) # Thrust uses cu, even for rocThrust
register_model(futhark FUTHARK FutharkStream.cpp)


set(USAGE ON CACHE BOOL "Whether to print all custom flags for the selected model")

message(STATUS "Available models:  ${REGISTERED_MODELS}")
if (NOT DEFINED MODEL)
    message(FATAL_ERROR "MODEL is unspecified, pick one from the available models")
else ()
    message(STATUS "Selected model  :  ${MODEL}")
endif ()

if (MODEL STREQUAL "sycl2020")
    message(FATAL_ERROR "
    Model sycl2020 has been renamed to sycl2020-acc, and a new sycl2020-usm model is now available.
    Please use sycl2020-acc for SYCL2020 style accessors and sycl2020-usm for USM")
endif ()

# load the $MODEL.cmake file and setup the correct IMPL_* based on $MODEL
load_model(${MODEL})

if (USAGE)
    # print the usage of the registered flag options
    registered_flags_action(print RESULT)
    message(STATUS "${RESULT}")
endif ()
# check required/set default for all registered flag options
registered_flags_action(check RESULT)
message(STATUS "${RESULT}")

# run model specific setup, i.e append build flags, etc
setup()

# CMake insists that -O2 (or equivalent) is the universally accepted optimisation level
# we remove that here and use our own <BUILD_TYPE>_FLAGS
wipe_gcc_style_optimisation_flags(CMAKE_CXX_FLAGS_${BUILD_TYPE})

message(STATUS "Default ${CMAKE_BUILD_TYPE} flags are `${DEFAULT_${BUILD_TYPE}_FLAGS}`, set ${BUILD_TYPE}_FLAGS to override (CXX_EXTRA_* flags are not affected)")

# setup common build flag defaults if there are no overrides
if (NOT DEFINED ${BUILD_TYPE}_FLAGS)
    set(ACTUAL_${BUILD_TYPE}_FLAGS ${DEFAULT_${BUILD_TYPE}_FLAGS})
else ()
    set(ACTUAL_${BUILD_TYPE}_FLAGS ${${BUILD_TYPE}_FLAGS})
endif ()


message(STATUS "CXX vendor  : ${CMAKE_CXX_COMPILER_ID} (${CMAKE_CXX_COMPILER})")
message(STATUS "Platform    : ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "Sources     : ${IMPL_SOURCES}")
message(STATUS "Libraries   : ${LINK_LIBRARIES}")
message(STATUS "CXX Flags   : ${CMAKE_CXX_FLAGS_${BUILD_TYPE}} ${ACTUAL_${BUILD_TYPE}_FLAGS} ${CXX_EXTRA_FLAGS}
    CXX flags derived from (CMake + (Override ? Override : Default) + Extras), where:
        CMake                    = `${CMAKE_CXX_FLAGS_${BUILD_TYPE}}`
        Default                  = `${DEFAULT_${BUILD_TYPE}_FLAGS}`
        Override (RELEASE_FLAGS) = `${${BUILD_TYPE}_FLAGS}`
        Extras (CXX_EXTRA_FLAGS) = `${CXX_EXTRA_FLAGS}`")
message(STATUS "Link Flags  : ${LINK_FLAGS} ${CXX_EXTRA_LINK_FLAGS}")
message(STATUS "Linker Flags: ${CMAKE_EXE_LINKER_FLAGS} ${CXX_EXTRA_LINKER_FLAGS} ")
message(STATUS "Defs        : ${IMPL_DEFINITIONS}")
message(STATUS "Executable  : ${EXE_NAME}")

# below we have all the usual CMake target setup steps

include_directories(src)
add_executable(${EXE_NAME} ${IMPL_SOURCES} src/main.cpp)
target_link_libraries(${EXE_NAME} PUBLIC ${LINK_LIBRARIES})
target_compile_definitions(${EXE_NAME} PUBLIC ${IMPL_DEFINITIONS})
target_include_directories(${EXE_NAME} PUBLIC ${IMPL_DIRECTORIES})

if (CXX_EXTRA_LIBRARIES)
    target_link_libraries(${EXE_NAME} PUBLIC ${CXX_EXTRA_LIBRARIES})
endif ()

target_compile_options(${EXE_NAME} PUBLIC "$<$<CONFIG:Release>:${ACTUAL_RELEASE_FLAGS};${CXX_EXTRA_FLAGS}>")
target_compile_options(${EXE_NAME} PUBLIC "$<$<CONFIG:Debug>:${ACTUAL_DEBUG_FLAGS};${CXX_EXTRA_FLAGS}>")

target_link_options(${EXE_NAME} PUBLIC LINKER:${CXX_EXTRA_LINKER_FLAGS})
target_link_options(${EXE_NAME} PUBLIC ${LINK_FLAGS} ${CXX_EXTRA_LINK_FLAGS})

# some models require the target to be already specified so they can finish their setup here
# this only happens if the model.cmake definition contains the `setup_target` macro
if (COMMAND setup_target)
    setup_target(${EXE_NAME})
endif ()

install(TARGETS ${EXE_NAME} DESTINATION bin)
