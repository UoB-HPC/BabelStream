
register_flag_optional(CMAKE_CXX_COMPILER
        "Any CXX compiler that supports OpenMP as per CMake detection (and offloading if enabled with `OFFLOAD`)"
        "c++")

register_flag_optional(ARCH
        "This overrides CMake's CMAKE_SYSTEM_PROCESSOR detection which uses (uname -p), this is mainly for use with
         specialised accelerators only and not to be confused with offload which is is mutually exclusive with this.
         Supported values are:
          - NEC"
        "")

register_flag_optional(OFFLOAD
        "Whether to use OpenMP offload, the format is <VENDOR:ARCH?>|ON|OFF.
        We support a small set of known offload flags for clang, gcc, and icpx.
        However, as offload support is rapidly evolving, we recommend you directly supply them via OFFLOAD_FLAGS.
        For example:
          * OFFLOAD=NVIDIA:sm_60
          * OFFLOAD=ON OFFLOAD_FLAGS=..."
        OFF)

register_flag_optional(OFFLOAD_FLAGS
        "If OFFLOAD is enabled, this *overrides* the default offload flags"
        "")

register_flag_optional(OFFLOAD_APPEND_LINK_FLAG
        "If enabled, this appends all resolved offload flags (OFFLOAD=<vendor:arch> or directly from OFFLOAD_FLAGS) to the link flags.
        This is required for most offload implementations so that offload libraries can linked correctly."
        ON)

#register_flag_optional(VECPAR_BACKEND
 #       "Valid values:
  #          * VECPAR_BACKEND=OFF (default)
   #         * VECPAR_BACKEND=CUDA
    #        * VECPAR_BACKEND=OMPT" OFF)

register_flag_optional(MEM "Device memory mode:
        DEFAULT   - allocate host and device memory pointers.
        MANAGED   - use CUDA Managed Memory"
        "DEFAULT")

set(VECPAR_FLAGS_CLANG -fopenmp)
set(VECPAR_FLAGS_OFFLOAD_CLANG_NVIDIA --language=cuda)
#set(VECPAR_BACKEND "CUDA")

macro(setup)

    set(CMAKE_CXX_STANDARD 20)
    set(LINKER_LANGUAGE CXX)
    register_definitions(${MEM})

    string(TOUPPER ${CMAKE_CXX_COMPILER_ID} COMPILER)
    find_package(vecpar REQUIRED 0.0.3)

    find_package(OpenMP QUIET)
    #find_package(vecmem QUIET)


    if (("${OFFLOAD}" STREQUAL OFF) OR (NOT DEFINED OFFLOAD))
        # no offload

        # CPU OpenMP backend can be built by either GCC or Clang
        register_link_library(OpenMP::OpenMP_CXX)

        list(APPEND VECPAR_FLAGS -fopenmp)
        # resolve the CPU specific flags
   #     set(VECPAR_FLAGS -fopenmp)
    #    set(VECPAR_LINK_FLAGS -fopenmp)
  #      register_append_compiler_and_arch_specific_cxx_flags(
   #             VECPAR_FLAGS_CPU
    #            ${COMPILER}
     #           ${ARCH}
     #   )

      #  register_append_compiler_and_arch_specific_link_flags(
       #         VECPAR_LINK_FLAGS_CPU
        #        ${COMPILER}
         #       ${ARCH}
        #)

    elseif ("${OFFLOAD}" STREQUAL ON)
        #  offload but with custom flags
        find_package(CUDAToolkit QUIET)
        register_definitions(VECPAR_GPU)
        separate_arguments(OFFLOAD_FLAGS)
        set(VECPAR_FLAGS ${OFFLOAD_FLAGS})
        register_link_library(CUDA::cudart)
        register_link_library(vecmem::cuda)
        register_link_library(vecpar::all)
    elseif ((DEFINED OFFLOAD) AND OFFLOAD_FLAGS)
        # offload but OFFLOAD_FLAGS overrides
        find_package(CUDAToolkit QUIET)
        register_definitions(VECPAR_GPU)
        separate_arguments(OFFLOAD_FLAGS)
        list(VECPAR_FLAGS APPEND  ${OFFLOAD_FLAGS})
        register_link_library(CUDA::cudart)
        register_link_library(vecmem::cuda)
        register_link_library(vecpar::all)
    else ()
        find_package(CUDAToolkit QUIET)
        register_definitions(VECPAR_GPU)
     #   list(APPEND VECPAR_FLAGS "-x cuda")

        # handle the vendor:arch value
        string(REPLACE ":" ";" OFFLOAD_TUPLE "${OFFLOAD}")

        list(LENGTH OFFLOAD_TUPLE LEN)
        if (LEN EQUAL 1)
            #  offload with <vendor> tuple
            list(GET OFFLOAD_TUPLE 0 OFFLOAD_VENDOR)
            # append VECPAR_FLAGS_OFFLOAD_<vendor> if  exists
            list(APPEND VECPAR_FLAGS ${VECPAR_FLAGS_OFFLOAD_${OFFLOAD_VENDOR}})

        elseif (LEN EQUAL 2)
            #  offload with <vendor:arch> tuple
            list(GET OFFLOAD_TUPLE 0 OFFLOAD_VENDOR)
            list(GET OFFLOAD_TUPLE 1 OFFLOAD_ARCH)

            # append VECPAR_FLAGS_OFFLOAD_<compiler>_<vendor> if exists
            list(APPEND VECPAR_FLAGS ${VECPAR_FLAGS_OFFLOAD_${COMPILER}_${OFFLOAD_VENDOR}})
            # append offload arch if VECPAR_FLAGS_OFFLOAD_<compiler>_ARCH_FLAG if exists
            if (DEFINED VECPAR_FLAGS_OFFLOAD_${COMPILER}_ARCH_FLAG)
                list(APPEND VECPAR_FLAGS
                        ${VECPAR_FLAGS_OFFLOAD_${COMPILER}_ARCH_FLAG}${OFFLOAD_ARCH})
            endif ()
            list(APPEND VECPAR_FLAGS --offload-arch=${OFFLOAD_ARCH})
        else ()
            message(FATAL_ERROR "Unrecognised OFFLOAD format: `${OFFLOAD}`, consider directly using OFFLOAD_FLAGS")
        endif ()

        register_link_library(CUDA::cudart)
        register_link_library(vecmem::cuda)
    endif ()


    register_link_library(vecpar::all)
    register_link_library(vecmem::core)

    # propagate flags to linker so that it links with the offload stuff as well
    register_append_cxx_flags(ANY ${VECPAR_FLAGS})
#    if (OFFLOAD_APPEND_LINK_FLAG)
 #       register_append_link_flags(${VECPAR_FLAGS})
  #  endif ()

endmacro()


