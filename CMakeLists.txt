cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

project(BabelStream VERSION 3.5 LANGUAGES CXX)

#set(CMAKE_VERBOSE_MAKEFILE ON)

# some nicer defaults for standard C++
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


#set(MODEL SYCL)
#set(SYCL_COMPILER COMPUTECPP)
#set(SYCL_COMPILER_DIR /home/tom/Desktop/computecpp_archive/ComputeCpp-CE-2.3.0-x86_64-linux-gnu)
#set(MODEL RAJA)
#set(RAJA_IN_TREE /home/tom/Downloads/RAJA-v0.13.0/)
#set(ENABLE_CUDA ON)
#set(TARGET NVIDIA)
#set(CUDA_TOOLKIT_ROOT_DIR /opt/cuda-11.2)
#set(CUDA_ARCH sm_70)
#set(BLT_DIR /home/tom/Downloads/blt-0.3.6/)

#set(MODEL STD)
#set(ARCH  cc70)
#set(CXX_EXTRA_FLAGS -v)

#set(MODEL CUDA)
#set(ARCH  sm_70)
#set(CMAKE_CUDA_COMPILER /opt/cuda-11.2/bin/nvcc)

#set(MODEL OCL)
#set(OpenCL_LIBRARY /opt/rocm-4.0.0/opencl/lib/libOpenCL.so)
#set(OpenCL_INCLUDE_DIR /opt/rocm-4.0.0/opencl/lib)
#set(RELEASE_FLAGS -Ofast)
#set(CXX_EXTRA_FLAGS -O2)

#set(CMAKE_CXX_COMPILER /usr/lib/aomp/bin/clang++)
#set(MODEL OMP)
##set(OFFLOAD "AMD:gfx803")
#set(OFFLOAD "NVIDIA:sm_35")
#set(CXX_EXTRA_FLAGS --cuda-path=/opt/cuda-10.2/)

#set(OFFLOAD "AMD:_70")
#set(CXX_EXTRA_FLAGS --cuda-path=/opt/cuda-10.2/ --gcc-toolchain=/home/tom/spack/opt/spack/linux-fedora33-zen2/gcc-10.2.1/gcc-8.3.0-latmjo2hl2yv53255xkwko7k3y7bx2vv)
#set(CXX_EXTRA_LINKER_FLAGS  )
#set(MODEL HIP)

#set(MODEL KOKKOS)
#set(KOKKOS_IN_TREE /home/tom/Downloads/kokkos-3.3.00/)

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

# copy CXX_EXTRA_FLAGS <- CXX_EXTRA_LINK_FLAGS
if ((DEFINED CXX_EXTRA_FLAGS) AND (NOT DEFINED CXX_EXTRA_LINK_FLAGS))
    set(CXX_EXTRA_LINK_FLAGS ${CXX_EXTRA_FLAGS})
endif ()

# include our macros
include(register_models.cmake)

# register out models <model_name> <preprocessor_def_name> <source files...>
register_model(OMP OMP OMPStream.cpp)
register_model(OCL OCL OCLStream.cpp)
register_model(STD STD STDStream.cpp)
register_model(STD20 STD20 STD20Stream.cpp)
register_model(HIP HIP HIPStream.cpp)
register_model(CUDA CUDA CUDAStream.cu)
register_model(KOKKOS KOKKOS KokkosStream.cpp)
register_model(SYCL SYCL SYCLStream.cpp)
register_model(ACC ACC ACCStream.cpp)
# defining RAJA collides with the RAJA namespace so USE_RAJA
register_model(RAJA USE_RAJA RAJAStream.cpp)
register_model(TBB TBB TBBStream.cpp)


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
    elseif()
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

add_executable(${EXE_NAME} ${IMPL_SOURCES} main.cpp)
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
# this only happens if the MODEL.cmake definition contains the `setup_target` macro
if (COMMAND setup_target)
    setup_target(${EXE_NAME})
endif ()

install (TARGETS ${EXE_NAME} DESTINATION bin)