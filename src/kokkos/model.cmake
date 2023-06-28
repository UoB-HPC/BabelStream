register_flag_optional(CMAKE_CXX_COMPILER
        "Any CXX compiler that is supported by CMake detection and RAJA.
         See https://github.com/kokkos/kokkos#primary-tested-compilers-on-x86-are"
        "c++")

register_flag_optional(KOKKOS_IN_TREE
        "Absolute path to the *source* distribution directory of Kokkos.
         Remember to append Kokkos specific flags as well, for example:
             -DKOKKOS_IN_TREE=... -DKokkos_ENABLE_OPENMP=ON -DKokkos_ARCH_ZEN=ON ...
         See https://github.com/kokkos/kokkos/blob/master/BUILD.md for all available options" "")

register_flag_optional(KOKKOS_IN_PACKAGE
        "Absolute path to package R-Path containing Kokkos libs.
         Use this instead of KOKKOS_IN_TREE if Kokkos is from a package manager like Spack." "")

# compiler vendor and arch specific flags
set(KOKKOS_FLAGS_CPU_INTEL -qopt-streaming-stores=always)

macro(setup)

    set(CMAKE_CXX_STANDARD 17) # Kokkos 4+ requires CXX >= 17
    cmake_policy(SET CMP0074 NEW) #see https://github.com/kokkos/kokkos/blob/master/BUILD.md


    if (EXISTS "${KOKKOS_IN_TREE}")
        message(STATUS "Build using in-tree Kokkos source at `${KOKKOS_IN_TREE}`")
        add_subdirectory(${KOKKOS_IN_TREE} ${CMAKE_BINARY_DIR}/kokkos)
        register_link_library(Kokkos::kokkos)
    elseif (EXISTS "${KOKKOS_IN_PACKAGE}")
        message(STATUS "Build using packaged Kokkos at `${KOKKOS_IN_PACKAGE}`")
        set (Kokkos_DIR "${KOKKOS_IN_PACKAGE}/lib64/cmake/Kokkos")
        find_package(Kokkos REQUIRED)
        register_link_library(Kokkos::kokkos)
    else()
        message(FATAL_ERROR "Neither `KOKKOS_IN_TREE`, or `KOKKOS_IN_PACKAGE` was set!")
    endif ()

    register_append_compiler_and_arch_specific_cxx_flags(
            KOKKOS_FLAGS_CPU
            ${CMAKE_CXX_COMPILER_ID}
            ${CMAKE_SYSTEM_PROCESSOR}
    )

endmacro()
