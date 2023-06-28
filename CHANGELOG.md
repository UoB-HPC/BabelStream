# Changelog
All notable changes to this project will be documented in this file.

## Unreleased
### Added
- Ability to build Kokkos and RAJA versions against existing packages.

### Changed
- RAJA CUDA CMake build issues resolved.
- Fix CUDA memory limit check.
- Use long double for `check_solution` in case of large problem size.
- OneAPI DPCPP compiler is deprecated in favour of ICPX, so added new build option to SYCL 2020 version.
- Updates to the HIP kernels and API usage.
- Number of thread-blocks in CUDA dot kernel implementation changed to 1024.

## [v4.0] - 2021-12-22

### Added
- New implementation using the C++ parallel STL (C++17).
- New implementation using C++20 range factories and `for_each_n`.
- Compiler options for OpenMP and OpenACC GNU offloading to NVIDIA and AMD.
- Compiler options for Arm Clang added to OpenMP and Kokkos.
- Kokkos 3 build system (No code changes made).
- SYCL build rules for ComputeCpp, DPCPP and HipSYCL.
- Support for CUDA Managed Memory and Page Fault memory.
- Added nstream kernel from PRK with associate command line option.
- CMake build system added for all models.
- SYCL device check for FP64 support.
- New implementations: TBB, Thrust, Julia, Scala, Java.
- Compiler options for Fujitsu added to OpenMP.

### Changed
- Default branch renamed from `master` to `main`.
- Array size is now a signed integer, which follows best practice.
- Driver now delays allocating large checking vectors until after computation has finished.
- Use cl::sycl::id parameters instead of cl::sycl::item.
- Update local copy of OpenCL C++ header file.
- Ensure correct SYCL queue constructor with explicit async_handler.
- Use built in SYCL runtime device discovery.
- Cray compiler OpenMP flags updated.
- Clang compiler OpenMP flags corrected for NVIDIA target.
- Reorder OpenCL objects in class so destructors are called in safe order.
- Ensure all OpenCL kernels are present in destructor.
- Unified run function in driver code to reduce code duplication, output should be uneffected.
- Normalise sum result by expected value to help false negative errors.
- HC version deprecated and moved to a legacy directory.
- Update RAJA to v0.13.0 (w/ code changes as this is a source incompatible update).
- Update SYCL version to SYCL 2020.

### Removed
- Pre-building of kernels in SYCL version to ensure compatibility with SYCL 1.2.1.
  Pre-building kernels is also not required, and shows no overhead as the first iteration is not timed.
- OpenACC Cray compiler flags.
- Build support for Kokkos 2.x (No code changes made).
- All Makefiles; build system will now use CMake exclusively.

## [v3.4] - 2019-04-10

### Added
- OpenACC flags to build for Power 9, Volta, Skylake and KNL.
- Kokkos list CLI argument shows some information about which device will be used.
- OpenMP GNU compiler now uses native target flag.
- Support CSV output for Triad only running mode.
- NEC and PGI compiler option for OpenMP version.
- Option to calculate memory bandwidth in base 2 (MiB/s) rather than base 10 (MB/s).

### Changed
- Update SYCL implementation to SYCL 1.2.1 interface.
- Output formatting of Kokkos implementation.
- Capitalisation of Kokkos filenames.
- Updated HIP implementation to new interface.
- Use parallel loop instead of kernels for OpenACC.
- OpenMP build for XL compiler uses `-qarch=auto`.

### Removed
- Superfluous OpenMP 4.5 map(to:) clauses on kernel target regions.
- Kokkos namespace not used by default so the API is easier to spot.
- Manual specification of Kokkos layout (DEVICE) as the Kokkos library sets this by default.

### Fixed
- Kokkos now compiles and links separately to fix complication with Kokkos 2.05.00.
- Kokkos can now instantiate single and double precision.
- OpenMP 4.5 map and reduction clause order to ensure reduction result copied back.
- Potential race condition in SYCL code between unloading OpenCL library and device list deconstructor.


## [v3.3] - 2017-12-04

### Added
- Add runtime option to run just the Triad kernel.
- Add runtime option for CSV output of results.
- ROCm HC implementation added for AMD GPUs.

### Changed
- Renamed project to BabelStream (from GPU-STREAM).
- Update SYCL Makefile to use ComputeCpp path variables.
- SYCL exceptions are now fatal, and are propagated to a runtime exception.


## [v3.2] - 2017-04-06

### Added
- Build instructions for RAJA and Kokkos libraries.

### Changed
- Use RAJA and Kokkos internal iterator types instead of int.
- Ensure RAJA pointers do not alias.
- Align memory to 2MB pages in RAJA and OpenMP.
- Updated Intel compiler flags for OpenMP, Kokkos and RAJA to ensure streaming stores.
- CUDA Makefile now uses variables to set compiler and flags.
- Use static shared memory for dot kernel in CUDA and HIP.

### Fixed
- Fix initialisation of b array bug in Kokkos implementation.


## [v3.1] - 2017-02-25

### Added
- Dot kernel HIP implementation.

### Changed
- Build system overhauled from CMake to a series of Makefiles.

### Deprecated
- Android build instructions.


## [v3.0] - 2017-01-30

### Added
- New Dot kernel added to the 4 standard kernels.

### Changed
- All model implementations now initialise and allocate their own arrays rather than copying from a master copy. This allows for better performance on NUMA architectures.
- Version string definition moved from header to main file.
- Combined OpenMP 3 and 4.5 implementations.
- OpenMP 4.5 target implementation uses alloc instead of to.
- Made SYCL indexing consistent.
- Update SYCL CMake build to use ComputeCpp CE 0.1.1.

### Fixed
- OpenMP deconstructor now only frees GPU memory only on GPU build.
- SYCL template specializations for float and double.


## [v2.1] - 2016-10-21

### Added
- New HIP version added.
- Results for v2.0 added.
- Output of OpenCL kernel build log on failure.

### Changed
- Use globally defined scalar value.
- Change scalar value to stop overflow.
- Restructure results directory.
- Change SYCL default work-group size.
- CMake defaults to Release build.

### Fixed
- CUDA device name output string corrected.
- Out of tree builds.


## [v2.0] - 2016-06-30

### Added
- Implementations in OpenMP 4.5 OpenACC, RAJA, Kokkos and SYCL.
- Copyright headers to source files.
- Runtime option variables are printed out.
- Device selection added to OpenCL and CUDA.

### Changed
- Major refactor to include multiple programming models. The change now uses C++ for driver code, with different models plugged in as classes which implement the STREAM kernels.
- Starting values in the arrays to reduce floating point errors with high iteration counts.
- Default array size now 2^25.
- Default to 100 iterations instead of 10.
- CUDA thread block size set via define rather than hardcoded value.
- Require CUDA 7 for C++11 support.
- OpenCL C++ header updated.

### Fixed
- Various CMake build fixes.
- Require at least 2 iterations.

### Removed
- Warning message for single precision iterations.


## [v1.1] - 2016-05-09

### Added
- HIP implementation and results.
- Titan X and Fury X results.
- Output of array sizes and other information at runtime.
- Ability to set CUDA block sizes on command line.
- Android build instructions.

### Changed
- Update OpenCL C++ header.
- Requires CUDA 6.5 or above.
- OpenCL uses Kernel Functor APIs instead of make_kernel API.

### Fixed
- Unsigned integer warnings.


## [v0.9] - 2015-08-07

Initial public release of OpenCL and CUDA GPU-STREAM.
