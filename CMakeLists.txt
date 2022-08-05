cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

project(BabelStream VERSION 4.0 LANGUAGES CXX)

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

# select default build type
set(CMAKE_BUILD_TYPE "Release")
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
register_model(sycl2020 SYCL2020 SYCLStream2020.cpp)
register_model(acc ACC ACCStream.cpp)
# defining RAJA collides with the RAJA namespace so USE_RAJA
register_model(raja USE_RAJA RAJAStream.cpp)
register_model(tbb TBB TBBStream.cpp)
register_model(thrust THRUST ThrustStream.cu) # Thrust uses cu, even for rocThrust


set(USAGE ON CACHE BOOL "Whether to print all custom flags for the selected model")

message(STATUS "Available models:  ${REGISTERED_MODELS}")
if (NOT DEFINED MODEL)
    message(FATAL_ERROR "MODEL is unspecified, pick one from the available models")
else ()
    message(STATUS "Selected model  :  ${MODEL}")
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
