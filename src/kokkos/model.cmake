
register_flag_optional(CMAKE_CXX_COMPILER
        "Any CXX compiler that is supported by CMake detection and RAJA.
         See https://github.com/kokkos/kokkos#primary-tested-compilers-on-x86-are"
        "c++")

register_flag_required(KOKKOS_IN_TREE
        "Absolute path to the *source* distribution directory of Kokkos.
         Remember to append Kokkos specific flags as well, for example:

             -DKOKKOS_IN_TREE=... -DKokkos_ENABLE_OPENMP=ON -DKokkos_ARCH_ZEN=ON ...

         See https://github.com/kokkos/kokkos/blob/master/BUILD.md for all available options")

# compiler vendor and arch specific flags
set(KOKKOS_FLAGS_CPU_INTEL -qopt-streaming-stores=always)

macro(setup)

    set(CMAKE_CXX_STANDARD 14)
    cmake_policy(SET CMP0074 NEW) #see https://github.com/kokkos/kokkos/blob/master/BUILD.md

    message(STATUS "Building using in-tree Kokkos source at `${KOKKOS_IN_TREE}`")

    if (EXISTS "${KOKKOS_IN_TREE}")
        add_subdirectory(${KOKKOS_IN_TREE} ${CMAKE_BINARY_DIR}/kokkos)
        register_link_library(Kokkos::kokkos)
    else ()
        message(FATAL_ERROR "`${KOKKOS_IN_TREE}` does not exist")
    endif ()

    register_append_compiler_and_arch_specific_cxx_flags(
            KOKKOS_FLAGS_CPU
            ${CMAKE_CXX_COMPILER_ID}
            ${CMAKE_SYSTEM_PROCESSOR}
    )

endmacro()


