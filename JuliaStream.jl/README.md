JuliaStream.jl
==============

This is an implementation of BabelStream in Julia which contains the following variants:

 * `PlainStream.jl` - Single threaded `for`
 * `ThreadedStream.jl` - Threaded implementation with `Threads.@threads` macros
 * `DistributedStream.jl` - Process based parallelism with `@distributed` macros
 * `CUDAStream.jl` - Direct port of BabelStream's native CUDA implementation using [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl)
 * `AMDGPUStream.jl` - Direct port of BabelStream's native HIP implementation using [AMDGPU.jl](https://github.com/JuliaGPU/AMDGPU.jl)

### Build & Run

Prerequisites

 * Julia >= 1.6+

With Julia on path, run the benchmark with:

```shell
> cd JuliaStream.jl
> julia --project -e 'import Pkg; Pkg.instantiate()' # only required on first run
> julia --project src/<IMPL>Stream.jl
```

**Important:**
 * Julia is 1-indexed, so N >= 1 in `--device N`.
 * Thread count for `ThreadedStream` must be set via the `JULIA_NUM_THREADS` environment variable (e.g `export JULIA_NUM_THREADS=$(nproc)`) otherwise it defaults to 1.
 * Worker count for `DistributedStream` is set with `-p <N>` as per the [documentation](https://docs.julialang.org/en/v1/manual/distributed-computing).
 * Certain implementations such as CUDA and AMDGPU will do hardware detection at runtime and may download and/or compile further software packages for the platform.
