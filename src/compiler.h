// (c) University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code
// 

#if (!defined(__TARGET_INCLUDED__))
// Expand a macro and convert the result into a string
#define STRINGIFY1(...) #__VA_ARGS__
#define STRINGIFY(...) STRINGIFY1(__VA_ARGS__)

// Work out the name of the compiler being used
#if defined(__cray__)
#define COMPILER_NAME "Cray " __VERSION__
#elif defined(__INTEL_COMPILER)
#define COMPILER_NAME                                                          \
  "Intel " STRINGIFY(__INTEL_COMPILER) "v" STRINGIFY(                         \
      __INTEL_COMPILER_BUILD_DATE)
#elif defined(__clang__)
#define COMPILER_NAME                                                          \
  "LLVM " STRINGIFY(__clang_major__) ":" STRINGIFY(                           \
      __clang_minor__) ":" STRINGIFY(__clang_patchlevel__)
#elif defined(__GNUC__)
#define COMPILER_NAME "GCC " __VERSION__
#else
#define COMPILER_NAME "Unknown compiler"
#endif
#define __TARGET_INCLUDED__
#endif // Include guard
