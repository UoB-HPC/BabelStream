# Spack Instructions

## Table of contents
* [OpenMP](#omp)
* [OpenCL](#ocl)
* [STD](#std)
* [STD20](#std20)
* [Hip](#hip)
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

| Flag        | Definition                      | Options   |
|-----------| ----------------------------------|-----------|
| cuda_arch     | List of supported compute capabilities are provided [here](https://github.com/spack/spack/blob/0f271883831bec6da3fc64c92eb1805c39a9f09a/lib/spack/spack/build_systems/cuda.py#LL19C1-L47C6) <br /> Useful [link](https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/) for matching CUDA gencodes with NVIDIA architectures| `cuda_arch=70` |
|amdgpu_target| List of supported architectures are provided [here](https://github.com/spack/spack/blob/0f271883831bec6da3fc64c92eb1805c39a9f09a/lib/spack/spack/build_systems/rocm.py#LL93C1-L125C19) | 'amdgpu_target=gfx701` |

Example Commandss
```shell
# Example 1: for Intel offload
 $ spack install babelstream%oneapi +omp 

# Example 2: for Nvidia GPU for Volta (sm_70) 
 $ spack install babelstream +omp cuda_arch=70 
 
# Example 3: for AMD GPU gfx701 
 $ spack install babelstream +omp amdgpu_target=gfx701 
```

