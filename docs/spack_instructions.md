# Spack Instructions


## Table of contents
* [OpenMP](#omp)
* [OpenCL](#ocl)
* [STD](#std)
* [Hip(ROCM)](#hip)
* [Cuda](#cuda)
* [Kokkos](#kokkos)
* [Sycl](#sycl)
* [Sycl2020](#)
* [ACC](#acc)
* [Raja](#raja)
* [Tbb](#tbb)
* [Thrust](#thrust)

## OpenMP

* There are 3 offloading options for OpenMP: NVIDIA, AMD and Intel. 
* If a user provides a value for `cuda_arch`, the execution will be automatically offloaded to NVIDIA.
* If a user provides a value for `amdgpu_target`, the operation will be offloaded to AMD.
* In the absence of `cuda_arch` and `amdgpu_target`, the execution will be offloaded to Intel.

| Flag        | Definition                      | 
|-----------| ----------------------------------|
| cuda_arch     |- List of supported compute capabilities are provided [here](https://github.com/spack/spack/blob/0f271883831bec6da3fc64c92eb1805c39a9f09a/lib/spack/spack/build_systems/cuda.py#LL19C1-L47C6) <br />- Useful [link](https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/) for matching CUDA gencodes with NVIDIA architectures| 
|amdgpu_target| List of supported architectures are provided [here](https://github.com/spack/spack/blob/0f271883831bec6da3fc64c92eb1805c39a9f09a/lib/spack/spack/build_systems/rocm.py#LL93C1-L125C19) | 


```shell
# Example 1: for Intel offload
 $ spack install babelstream%oneapi +omp 

# Example 2: for Nvidia GPU for Volta (sm_70) 
 $ spack install babelstream +omp cuda_arch=70 
 
# Example 3: for AMD GPU gfx701 
 $ spack install babelstream +omp amdgpu_target=gfx701 
```


## OpenCL

* No need to specify `amdgpu_target` or `cuda_arch` here since we are using AMD and CUDA as backend respectively.


| Flag        | Definition                      | 
|-----------| ----------------------------------|
| backend     | 4 different backend options: <br />- cuda <br />- amd <br />- intel <br />- pocl | 


```shell
# Example 1:  CUDA backend
 $ spack install babelstream%gcc +ocl backend=cuda

# Example 2:  AMD backend 
 $ spack install babelstream%gcc +ocl backend=amd
 
# Example 3:  Intel backend
 $ spack install babelstream%gcc +ocl backend=intel

# Example 4:  POCL backend
 $ spack install babelstream%gcc +ocl backend=pocl
```

## STD
* Minimum GCC version requirement `10.1.0`
* NVHPC Offload will be added in the future release 

```shell
# Example 1:  data 
 $ spack install babelstream +stddata

# Example 2:  ranges
 $ spack install babelstream +stdranges
 
# Example 3:  indices
 $ spack install babelstream +stdindices

```

## HIP(ROCM)

*  `amdgpu_target` and `flags` are optional here.


| Flag        | Definition                      | 
|-----------| ----------------------------------|
|amdgpu_target| List of supported architectures are provided [here](https://github.com/spack/spack/blob/0f271883831bec6da3fc64c92eb1805c39a9f09a/lib/spack/spack/build_systems/rocm.py#LL93C1-L125C19) | 
|flags | Extra flags to pass |



```shell
# Example 1:  ROCM default
 $ spack install babelstream +rocm

# Example 2:  ROCM with GPU target
 $ spack install babelstream +rocm amdgpu_target=<gfx701>
 
# Example 3:  ROCM with extra flags option
 $ spack install babelstream +rocm flags=<xxx>

# Example 4:  ROCM with GPU target and extra flags
 $ spack install babelstream +rocm amdgpu_target=<gfx701> flags=<xxx>
```

## CUDA

* The `cuda_arch` value is mandatory here. 
* If a user provides a value for `mem`, device memory mode will be chosen accordingly
* If a user provides a value for `flags`, additional CUDA flags will be passed to NVCC
* In the absence of `mem` and `flags`, the execution will choose **DEFAULT** for device memory mode and no additional flags will be passed


| Flag        | Definition                      | 
|-----------| ----------------------------------|
| cuda_arch     |- List of supported compute capabilities are provided [here](https://github.com/spack/spack/blob/0f271883831bec6da3fc64c92eb1805c39a9f09a/lib/spack/spack/build_systems/cuda.py#LL19C1-L47C6) <br />- Useful [link](https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/) for matching CUDA gencodes with NVIDIA architectures| 
|mem| Device memory mode: <br />- **DEFAULT** allocate host and device memory pointers.<br />- **MANAGED** use CUDA Managed Memory.<br />- **PAGEFAULT** shared memory, only host pointers allocated | 
|flags | Extra flags to pass |

```shell
# Example 1: CUDA no mem and flags specified
 $ spack install babelstream +cuda cuda_arch=<70>

# Example 2: for Nvidia GPU for Volta (sm_70) 
 $ spack install babelstream +cuda cuda_arch=<70> mem=<managed>
 
# Example 3: CUDA with mem and flags specified
 $ spack install babelstream +cuda cuda_arch=<70> mem=<managed> flags=<CUDA_EXTRA_FLAGS> 
```

## Kokkos

* Kokkos implementation requires kokkos source folder to be provided because it builds it from the scratch


| Flag        | Definition                      | 
|-----------| ----------------------------------|
| dir | Download the kokkos release from github repository ( https://github.com/kokkos/kokkos ) and extract the zip file to a directory you want and target this directory with `dir` flag |
| backend     | 2 different backend options: <br />- cuda <br />- omp | 
| cuda_arch     |- List of supported compute capabilities are provided [here](https://github.com/spack/spack/blob/0f271883831bec6da3fc64c92eb1805c39a9f09a/lib/spack/spack/build_systems/cuda.py#LL19C1-L47C6) <br />- Useful [link](https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/) for matching CUDA gencodes with NVIDIA architectures| 


```shell
# Example 1:  No Backend option specified
 $ spack install babelstream +kokkos dir=</home/user/Downloads/kokkos-x.x.xx>

# Example 2:  CUDA backend 
 $ spack install babelstream +kokkos backend=cuda cuda_arch=70 dir=</home/user/Downloads/kokkos-x.x.xx>
 
# Example 3:  OMP backend
 $ spack install babelstream +kokkos  backend=omp dir=</home/user/Downloads/kokkos-x.x.xx>

```


## SYCL2020
* Instructions for installing the intel compilers are provided [here](https://spack.readthedocs.io/en/latest/build_systems/inteloneapipackage.html#building-a-package-with-icx)

| Flag        | Definition                      | 
|-----------| ----------------------------------|
| implementation     | 3 different implementation options: <br />- OneAPI-ICPX <br />- OneAPI-DPCPP <br />- Compute-CPP <br />| 

```shell
# Example 1:  No implementation option specified (build for OneAPI-ICPX)
 $ spack install babelstream%oneapi +sycl2020

# Example 2:  OneAPI-DPCPP implementation 
 $ spack install babelstream +sycl2020 implementation=ONEAPI-DPCPP
```

## SYCL

| Flag        | Definition                      | 
|-----------| ----------------------------------|
| implementation     | 2 different implementation options: <br />- OneAPI-DPCPP <br />- Compute-CPP <br />| 

```shell
# Example 1:  OneAPI-DPCPP implementation 
 $ spack install babelstream +sycl2020 implementation=ONEAPI-DPCPP
```
## ACC
* Target device selection process is automatic with 2 options:
    * **gpu** : Globally set the target device to an NVIDIA GPU automatically if `cuda_arch` is specified 
    * **multicore** : Globally set the target device to the host CPU automatically if `cpu_arch` is specified 

| Flag        | Definition                      | 
|-----------| ----------------------------------|
| cuda_arch     |- List of supported compute capabilities are provided [here](https://github.com/spack/spack/blob/0f271883831bec6da3fc64c92eb1805c39a9f09a/lib/spack/spack/build_systems/cuda.py#LL19C1-L47C6) <br />- Useful [link](https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/) for matching CUDA gencodes with NVIDIA architectures| 
| CPU_ARCH   | This sets the `-tp` (target processor) flag, possible values are: <br /> `px`          - Generic x86 Processor <br /> `bulldozer`   - AMD Bulldozer processor <br /> `piledriver`  - AMD Piledriver processor <br /> `zen`         - AMD Zen architecture (Epyc, Ryzen) <br /> `zen2`        - AMD Zen 2 architecture (Ryzen 2) <br />  `sandybridge` - Intel SandyBridge processor <br /> `haswell`     - Intel Haswell processor <br /> `knl`         - Intel Knights Landing processor <br /> `skylake`     - Intel Skylake Xeon processor <br /> `host`        - Link native version of HPC SDK cpu math library <br /> `native`      - Alias for -tp host | `cpu_arch=skylake` |

```shell
# Example 1:  For GPU Run 
 $ spack install babelstream +acc cuda_arch=<70>

# Example 2:  For Multicore CPU Run 
 $ spack install babelstream +acc cpu_arch=<bulldozer>
```

## RAJA
* RAJA implementation requires RAJA source folder to be provided because it builds it from the scratch


| Flag        | Definition                      | 
|-----------| ----------------------------------|
| dir | Download the Raja release from github repository and extract the zip file to a directory you want and target this directory with `dir` flag |
| backend     | 2 different backend options: <br />- cuda <br />- omp | 
|offload| Choose offloading platform `offload= [cpu]/[nvidia]` |

```shell
# Example 1:  For CPU offload with backend OMP 
 $ spack install babelstream +raja offload=cpu backend=omp dir=/home/dir/raja
```

## TBB
```shell
# Example: 
 $ spack install babelstream +tbb
```

## THRUST

| Flag        | Definition                      | 
|-----------| ----------------------------------|
|implementation| Choose one of the implementation for Thrust. Options are `cuda` and `rocm` | `implementation = [cuda]/[rocm]` |
|backend| CUDA's Thrust implementation supports the following backends:- CUDA- OMP - TBB |
| cuda_arch     |- List of supported compute capabilities are provided [here](https://github.com/spack/spack/blob/0f271883831bec6da3fc64c92eb1805c39a9f09a/lib/spack/spack/build_systems/cuda.py#LL19C1-L47C6) <br />- Useful [link](https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/) for matching CUDA gencodes with NVIDIA architectures| 
|flags | Additional CUDA flags passed to nvcc, this is appended after `CUDA_ARCH` | 

```shell
# Example1: CUDA implementation
$ spack install babelstream +thrust implementation=cuda backend=cuda cuda_arch=<70> flags=<option>
$ spack install babelstream +thrust implementation=cuda backend=omp cuda_arch=<70> flags=<option>
$ spack install babelstream +thrust implementation=cuda backend=tbb cuda_arch=<70> flags=<option>
# Example1: ROCM implementation
*  spack install babelstream +thrust implementation=rocm backend=<option>
```
