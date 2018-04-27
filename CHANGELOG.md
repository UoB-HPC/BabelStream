# Changelog
All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- OpenACC flags to build for Volta.
- Kokkos list CLI argument shows some information about which device will be used.
- OpenMP GNU compiler now uses native target flag

### Changed
- Update SYCL implementation to SYCL 1.2.1 interface.
- Output formatting of Kokkos implementation.
- Capitalisation of Kokkos filenames.
- Updated HIP implementation to new interface.

### Removed
- Superfluous OpenMP 4.5 map(to:) clauses on kernel target regions.
- Kokkos namespace not used by default so the API is easier to spot.
- Manual specification of Kokkos layout (DEVICE) as the Kokkos library sets this by default.

### Fixed
- Kokkos now compiles and links separately to fix complication with Kokkos 2.05.00.
- Kokkos can now instantiate single and double precision.
- OpenMP 4.5 map and reduction clause order to ensure reduction result copied back.


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
