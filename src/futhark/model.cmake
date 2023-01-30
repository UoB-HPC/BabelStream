# Use
#
#   cmake -Bbuild -H. -DMODEL=futhark -DFUTHARK_BACKEND=foo -DFUTHARK_COMPILER=foo/bar/bin/futhark
#
# to use the Futhark backend, where 'foo' must be one of 'multicore',
# 'c', 'opencl', or 'cuda'.  Defaults to 'multicore'.
#
# Use -DFUTHARK_COMPILER to set the path to the Futhark compiler
# binary.  Defaults to 'futhark' on the PATH.

register_flag_optional(FUTHARK_BACKEND
  "Use a specific Futhark backend, possible options are:
         - c
         - multicore
         - opencl
         - cuda"
  "multicore")

register_flag_optional(FUTHARK_COMPILER
  "Absolute path to the Futhark compiler, defaults to the futhark compiler on PATH"
  "futhark")

macro(setup)
  add_custom_command(
    OUTPUT
    ${CMAKE_CURRENT_BINARY_DIR}/babelstream.c
    ${CMAKE_CURRENT_BINARY_DIR}/babelstream.h
    COMMAND ${FUTHARK_COMPILER} ${FUTHARK_BACKEND}
    --library src/futhark/babelstream.fut
    -o ${CMAKE_CURRENT_BINARY_DIR}/babelstream
    DEPENDS src/futhark/babelstream.fut
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    VERBATIM
  )
  if (${FUTHARK_BACKEND} STREQUAL "c")
    # Nothing to do.
  elseif (${FUTHARK_BACKEND} STREQUAL "multicore")
    set(THREADS_PREFER_PTHREAD_FLAG ON)
    find_package(Threads REQUIRED)
    register_link_library(Threads::Threads)
  elseif (${FUTHARK_BACKEND} STREQUAL "opencl")
    find_package(OpenCL REQUIRED)
    register_link_library(OpenCL::OpenCL)
  elseif (${FUTHARK_BACKEND} STREQUAL "cuda")
    find_package(CUDA REQUIRED)
    register_link_library("nvrtc" "cuda" "cudart")
  else ()
    message(FATAL_ERROR "Unsupported Futhark backend: ${FUTHARK_BACKEND}")
  endif()
endmacro()

macro(setup_target)
  target_sources(${EXE_NAME} PUBLIC "${CMAKE_CURRENT_BINARY_DIR}/babelstream.c")
  include_directories("${CMAKE_CURRENT_BINARY_DIR}")
endmacro()
