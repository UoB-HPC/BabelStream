JuliaStream.jl
==============

This is an implementation of BabelStream in Julia which contains the following variants:

 * `PlainStream.jl` - Single threaded `for`
 * `ThreadedStream.jl` - Threaded implementation with `Threads.@threads` macros
 * `DistributedStream.jl` - Process based parallelism with `@distributed` macros
 * `CUDAStream.jl` - Direct port of BabelStream's native CUDA implementation using [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl)
 * `AMDGPUStream.jl` - Direct port of BabelStream's native HIP implementation using [AMDGPU.jl](https://github.com/JuliaGPU/AMDGPU.jl)
 * `oneAPIStream.jl` - Direct port of BabelStream's native SYCL implementation using [oneAPI.jl](https://github.com/JuliaGPU/oneAPI.jl)
 * `KernelAbstractions.jl` - Direct port of miniBUDE's native CUDA implementation using [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl)

### Build & Run

Prerequisites

 * Julia >= 1.6+

A set of reduced dependency projects are available for the following backend and implementations:

 * `AMDGPU` supports:
   - `AMDGPUStream.jl`
 * `CUDA` supports:
   - `CUDAStream.jl`
 * `oneAPI` supports:
   - `oneAPIStream.jl`
 * `KernelAbstractions` supports:
   - `KernelAbstractionsStream.jl`
 * `Threaded` supports:
   - `PlainStream.jl`
   - `ThreadedStream.jl`
   - `DistributedStream.jl`

With Julia on path, run your selected benchmark with:

```shell
> cd JuliaStream.jl
> julia --project=<BACKEND> -e 'import Pkg; Pkg.instantiate()' # only required on first run
> julia --project=<BACKEND> src/<IMPL>Stream.jl
```

For example. to run the CUDA implementation:

```shell
> cd JuliaStream.jl
> julia --project=CUDA -e 'import Pkg; Pkg.instantiate()' 
> julia --project=CUDA src/CUDAStream.jl
```

**Important:**
 * Julia is 1-indexed, so N >= 1 in `--device N`.
 * Thread count for `ThreadedStream` must be set via the `JULIA_NUM_THREADS` environment variable (e.g `export JULIA_NUM_THREADS=$(nproc)`) otherwise it defaults to 1.
 * Worker count for `DistributedStream` is set with `-p <N>` as per the [documentation](https://docs.julialang.org/en/v1/manual/distributed-computing).
 * Certain implementations such as CUDA and AMDGPU will do hardware detection at runtime and may download and/or compile further software packages for the platform.

***

Alternatively, the top-level project `Project.toml` contains all dependencies needed to run all implementations in `src`.
There may be instances where some packages are locked to an older version because of transitive dependency requirements.

To run the benchmark using the top-level project, run the benchmark with:
```shell
> cd JuliaStream.jl
> julia --project -e 'import Pkg; Pkg.instantiate()'  
> julia --project src/<IMPL>Stream.jl
```