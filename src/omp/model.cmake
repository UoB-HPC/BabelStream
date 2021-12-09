# Compiler ID for reference (as of CMake 3.13)
#    Absoft = Absoft Fortran (absoft.com)
#    ADSP = Analog VisualDSP++ (analog.com)
#    AppleClang = Apple Clang (apple.com)
#    ARMCC = ARM Compiler (arm.com)
#    Bruce = Bruce C Compiler
#    CCur = Concurrent Fortran (ccur.com)
#    Clang = LLVM Clang (clang.llvm.org)
#    Cray = Cray Compiler (cray.com)
#    Embarcadero, Borland = Embarcadero (embarcadero.com)
#    G95 = G95 Fortran (g95.org)
#    GNU = GNU Compiler Collection (gcc.gnu.org)
#    HP = Hewlett-Packard Compiler (hp.com)
#    IAR = IAR Systems (iar.com)
#    Intel = Intel Compiler (intel.com)
#    MIPSpro = SGI MIPSpro (sgi.com)
#    MSVC = Microsoft Visual Studio (microsoft.com)
#    NVIDIA = NVIDIA CUDA Compiler (nvidia.com)
#    OpenWatcom = Open Watcom (openwatcom.org)
#    PGI = The Portland Group (pgroup.com)
#    Flang = Flang Fortran Compiler
#    PathScale = PathScale (pathscale.com)
#    SDCC = Small Device C Compiler (sdcc.sourceforge.net)
#    SunPro = Oracle Solaris Studio (oracle.com)
#    TI = Texas Instruments (ti.com)
#    TinyCC = Tiny C Compiler (tinycc.org)
#    XL, VisualAge, zOS = IBM XL (ibm.com)

# These are only added in CMake 3.15:
#    ARMClang = ARM Compiler based on Clang (arm.com)
# These are only added in CMake 3.20:
#    NVHPC = NVIDIA HPC SDK Compiler (nvidia.com)
# These are only added in CMake 3.21
#    Fujitsu = Fujitsu HPC compiler (Trad mode)
#    FujitsuClang = Fujitsu HPC compiler (Clang mode)


# CMAKE_SYSTEM_PROCESSOR is set via `uname -p`, we have:
# Power9 = ppc64le
# x64    = x86_64
# arm64  = aarch64
#


# predefined offload flags based on compiler id and vendor,
# the format is (COMPILER and VENDOR must be UPPERCASE):
# Compiler: OMP_FLAGS_OFFLOAD_<COMPILER?>_<VNEDOR?>

set(OMP_FLAGS_OFFLOAD_INTEL
        -qnextgen -fiopenmp -fopenmp-targets=spir64)
set(OMP_FLAGS_OFFLOAD_GNU_NVIDIA
        -foffload=nvptx-none)
set(OMP_FLAGS_OFFLOAD_GNU_AMD
        -foffload=amdgcn-amdhsa)
set(OMP_FLAGS_OFFLOAD_CLANG_NVIDIA
        -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target=nvptx64-nvidia-cuda)
set(OMP_FLAGS_OFFLOAD_CLANG_AMD
        -fopenmp=libomp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa)
set(OMP_FLAGS_OFFLOAD_CLANG_ARCH_FLAG
        -march=) # prefix only, arch appended by the vendor:arch tuple


# for standard (non-offload) omp, the format is (COMPILER and ARCH must be UPPERCASE):
# Compiler:      OMP_FLAGS_CPU_<COMPILER?>_<ARCH?>
# Linker:   OMP_LINK_FLAGS_CPU_<COMPILER?>_<ARCH?>

set(OMP_FLAGS_CPU_FUJITSU
        -Kfast -std=c++11 -KA64FX -KSVE -KARMV8_3_A -Kzfill=100 -Kprefetch_sequential=soft -Kprefetch_line=8 -Kprefetch_line_L2=16)
set(OMP_LINK_FLAGS_CPU_FUJITSU
        -Kopenmp)

set(OMP_FLAGS_CPU_INTEL
        -qopt-streaming-stores=always)

set(OMP_FLAGS_CPU_GNU_PPC64LE
        -mcpu=native)

set(OMP_FLAGS_CPU_XL
        -O5 -qarch=auto -qtune=auto)

set(OMP_FLAGS_CPU_NEC -O4 -finline) # CMake doesn't detect this so it's meant to be chosen by register_flag_optional(ARCH)

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
          * OFFLOAD=AMD:gfx906
          * OFFLOAD=INTEL
          * OFFLOAD=ON OFFLOAD_FLAGS=..."
        OFF)

register_flag_optional(OFFLOAD_FLAGS
        "If OFFLOAD is enabled, this *overrides* the default offload flags"
        "")

register_flag_optional(OFFLOAD_APPEND_LINK_FLAG
        "If enabled, this appends all resolved offload flags (OFFLOAD=<vendor:arch> or directly from OFFLOAD_FLAGS) to the link flags.
        This is required for most offload implementations so that offload libraries can linked correctly."
        ON)


macro(setup)
    find_package(OpenMP REQUIRED)
    register_link_library(OpenMP::OpenMP_CXX)

    string(TOUPPER ${CMAKE_CXX_COMPILER_ID} COMPILER)
    if(NOT ARCH)
        string(TOUPPER ${CMAKE_SYSTEM_PROCESSOR} ARCH)
    else()
        message(STATUS "Using custom arch: ${ARCH}")
    endif()



    if (("${OFFLOAD}" STREQUAL OFF) OR (NOT DEFINED OFFLOAD))
        # no offload

        # resolve the CPU specific flags
        # starting with ${COMPILER_VENDOR}_${PLATFORM_ARCH}, then try ${COMPILER_VENDOR}, and then give up
        register_append_compiler_and_arch_specific_cxx_flags(
                OMP_FLAGS_CPU
                ${COMPILER}
                ${ARCH}
        )

        register_append_compiler_and_arch_specific_link_flags(
                OMP_LINK_FLAGS_CPU
                ${COMPILER}
                ${ARCH}
        )

    elseif ("${OFFLOAD}" STREQUAL ON)
        #  offload but with custom flags
        register_definitions(OMP_TARGET_GPU)
        separate_arguments(OFFLOAD_FLAGS)
        set(OMP_FLAGS ${OFFLOAD_FLAGS})
    elseif ((DEFINED OFFLOAD) AND OFFLOAD_FLAGS)
        # offload but OFFLOAD_FLAGS overrides
        register_definitions(OMP_TARGET_GPU)
        separate_arguments(OFFLOAD_FLAGS)
        list(OMP_FLAGS APPEND ${OFFLOAD_FLAGS})
    else ()

        # handle the vendor:arch value
        string(REPLACE ":" ";" OFFLOAD_TUPLE "${OFFLOAD}")

        list(LENGTH OFFLOAD_TUPLE LEN)
        if (LEN EQUAL 1)
            #  offload with <vendor> tuple
            list(GET OFFLOAD_TUPLE 0 OFFLOAD_VENDOR)
            # append OMP_FLAGS_OFFLOAD_<vendor> if  exists
            list(APPEND OMP_FLAGS ${OMP_FLAGS_OFFLOAD_${OFFLOAD_VENDOR}})

        elseif (LEN EQUAL 2)
            #  offload with <vendor:arch> tuple
            list(GET OFFLOAD_TUPLE 0 OFFLOAD_VENDOR)
            list(GET OFFLOAD_TUPLE 1 OFFLOAD_ARCH)

            # append OMP_FLAGS_OFFLOAD_<compiler>_<vendor> if exists
            list(APPEND OMP_FLAGS ${OMP_FLAGS_OFFLOAD_${COMPILER}_${OFFLOAD_VENDOR}})
            # append offload arch if OMP_FLAGS_OFFLOAD_<compiler>_ARCH_FLAG if exists
            if (DEFINED OMP_FLAGS_OFFLOAD_${COMPILER}_ARCH_FLAG)
                list(APPEND OMP_FLAGS
                        "${OMP_FLAGS_OFFLOAD_${COMPILER}_ARCH_FLAG}${OFFLOAD_ARCH}")
            endif ()
        else ()
            message(FATAL_ERROR "Unrecognised OFFLOAD format: `${OFFLOAD}`, consider directly using OFFLOAD_FLAGS")
        endif ()

    endif ()


    message(STATUS "OMP CXX  flags : ${OMP_FLAGS}")
    message(STATUS "OMP Link flags : ${OMP_LINK_FLAGS}")
    # propagate flags to linker so that it links with the offload stuff as well
    register_append_cxx_flags(ANY ${OMP_FLAGS})
    if (OFFLOAD_APPEND_LINK_FLAG)
        register_append_link_flags(${OMP_FLAGS})
    endif ()
endmacro()

