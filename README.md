# BabelStream

<img src="babelstream.png?raw=true" alt="logo" height="300" align="right" />

[![CI](https://github.com/UoB-HPC/BabelStream/actions/workflows/main.yaml/badge.svg?branch=main)](https://github.com/UoB-HPC/BabelStream/actions/workflows/main.yaml)

Measure memory transfer rates to/from global device memory on GPUs.
This benchmark is similar in spirit, and based on, the STREAM benchmark [1] for CPUs.

Unlike other GPU memory bandwidth benchmarks this does *not* include the PCIe transfer time.

There are multiple implementations of this benchmark in a variety of [programming models](#models).

This code was previously called GPU-STREAM.

## Table of Contents
- [Programming Models](#programming-models)
- [How is this different to STREAM?](#how-is-this-different-to-stream)
- [Building](#building)
    - [CMake](#cmake)
    - [Spack](#spack)
    - [GNU Make (removed)](#gnu-make)
- [Results](#results)
- [Contributing](#contributing)
- [Citing](#citing)
    - [Other BabelStream publications](#other-babelstream-publications)


## Programming Models

BabelStream is currently implemented in the following parallel programming models, listed in no particular order:

- OpenCL
- CUDA
- HIP
- OpenACC
- OpenMP 3 and 4.5
- C++ Parallel STL
- Kokkos
- RAJA
- SYCL and SYCL2020 (USM and accessors)
- TBB
- Thrust (via CUDA or HIP)
- Futhark

This project also contains implementations in alternative languages with different build systems:
* Julia - [JuliaStream.jl](./src/julia/JuliaStream.jl)
* Java - [java-stream](./src/java/java-stream)
* Scala - [scala-stream](./src/scala/scala-stream)
* Rust - [rust-stream](./src/rust/rust-stream)

## How is this different to STREAM?

BabelStream implements the four main kernels of the STREAM benchmark (along with a dot product), but by utilising different programming models expands the platforms which the code can run beyond CPUs.

The key differences from STREAM are that:
* the arrays are allocated on the heap
* the problem size is unknown at compile time
* wider platform and programming model support

With stack arrays of known size at compile time, the compiler is able to align data and issue optimal instructions (such as non-temporal stores, remove peel/remainder vectorisation loops, etc.).
But this information is not typically available in real HPC codes today, where the problem size is read from the user at runtime.

BabelStream therefore provides a measure of what memory bandwidth performance can be attained (by a particular programming model) if you follow today's best parallel programming best practice.

BabelStream also includes the nstream kernel from the Parallel Research Kernels (PRK) project, available on [GitHub](https://github.com/ParRes/Kernels).
Details about PRK can be found in the following references:

* Van der Wijngaart, Rob F., and Timothy G. Mattson. The parallel research kernels. IEEE High Performance Extreme Computing Conference (HPEC). IEEE, 2014.

* R. F. Van der Wijngaart, A. Kayi, J. R. Hammond, G. Jost, T. St. John, S. Sridharan, T. G. Mattson, J. Abercrombie, and J. Nelson. Comparing runtime systems with exascale ambitions using the Parallel Research Kernels. ISC 2016, [DOI: 10.1007/978-3-319-41321-1_17](https://doi.org/10.1007/978-3-319-41321-1_17).

* Jeff R. Hammond and Timothy G. Mattson. Evaluating data parallelism in C++ using the Parallel Research Kernels. IWOCL 2019, [DOI: 10.1145/3318170.3318192](https://doi.org/10.1145/3318170.3318192).


## Building

Drivers, compiler and software applicable to whichever implementation you would like to build against is required.

### CMake

The project supports building with CMake >= 3.13.0, which can be installed without root via the [official script](https://cmake.org/download/).

Each BabelStream implementation (programming model) is built as follows:

```shell
$ cd babelstream

# configure the build, build type defaults to Release
# The -DMODEL flag is required
$ cmake -Bbuild -H. -DMODEL=<model> <model specific flags prefixed with -D...>

# compile
$ cmake --build build

# run executables in ./build
$ ./build/<model>-stream
```

The `MODEL` option selects one implementation of BabelStream to build.
The source for each model's implementations are located in `./src/<model>`.

Currently available models are:
```
omp;ocl;std-data;std-indices;std-ranges;hip;cuda;kokkos;sycl;sycl2020-acc;sycl2020-usm;acc;raja;tbb;thrust;futhark
```

#### Overriding default flags
By default, we have defined a set of optimal flags for known HPC compilers.
There are assigned those to `RELEASE_FLAGS`, and you can override them if required.

To find out what flag each model supports or requires, simply configure while only specifying the model.
For example:
```shell
> cd babelstream
> cmake -Bbuild -H. -DMODEL=ocl 
...
- Common Release flags are `-O3`, set RELEASE_FLAGS to override
-- CXX_EXTRA_FLAGS: 
        Appends to common compile flags. These will be used at link phase at well.
        To use separate flags at link time, set `CXX_EXTRA_LINKER_FLAGS`
-- CXX_EXTRA_LINK_FLAGS: 
        Appends to link flags which appear *before* the objects.
        Do not use this for linking libraries, as the link line is order-dependent
-- CXX_EXTRA_LIBRARIES: 
        Append to link flags which appears *after* the objects.
        Use this for linking extra libraries (e.g `-lmylib`, or simply `mylib`) 
-- CXX_EXTRA_LINKER_FLAGS: 
        Append to linker flags (i.e GCC's `-Wl` or equivalent)
-- Available models:  omp;ocl;std;std20;hip;cuda;kokkos;sycl;acc;raja;tbb
-- Selected model  :  ocl
-- Supported flags:

   CMAKE_CXX_COMPILER (optional, default=c++): Any CXX compiler that is supported by CMake detection
   OpenCL_LIBRARY (optional, default=): Path to OpenCL library, usually called libOpenCL.so
...
```

Alternatively, refer to the [CI script](./src/ci-test-compile.sh), which test-compiles most of the models, and see which flags are used there.

*It is recommended that you delete the `build` directory when you change any of the build flags.*

### Spack


The project supports building with Spack >= 0.19.0, which can be installed without root via the [official GitHub repo](https://github.com/spack/spack).
The BabelStream Spack Package source code could be accessed from the link [here](https://github.com/spack/spack/tree/develop/var/spack/repos/builtin/packages/babelstream/package.py).
Each BabelStream implementation (programming model) is built as follows:

```shell

# Spack package installation starts with `spack install babelstream` for all programming models
# The programming model wish to be build needs to be specified with `+` option
# The model specific flags needs to be specified after defining model
$ spack install babelstream@<version>%<compiler> +<model> <model specific flags>


# The executables will be generated in:
# SPACK_INSTALL_DIRECTORY/opt/spack/system-name/compiler-name/babelstream-version-identifier/bin/
# this address will be printed at the end of generation which could be easily copied
$ cd SPACK_INSTALL_DIRECTORY/opt/spack/system-name/compiler-name/babelstream-version-identifier/bin/
$ ./<model>-stream
```
More detailed examples are provided in [Spack README file](./docs/spack_instructions.md).
The `MODEL` variant selects one implementation of BabelStream to build.

Currently available models are:
```
omp;ocl;std-data;std-indices;std-ranges;hip;cuda;kokkos;sycl;sycl2020-acc;sycl2020-usm;acc;raja;tbb;thrust
```

### GNU Make

Support for Make has been removed from 4.0 onwards.
However, as the build process only involves a few source files, the required compile commands can be extracted from the CI output.

<!-- TODO add CI snipped here -->

## Results

Sample results can be found in the `results` subdirectory.
Newer results are found in our [Performance Portability](https://github.com/UoB-HPC/performance-portability) repository.


## Contributing

As of v4.0, the `main` branch of this repository will hold the latest released version.

The `develop` branch will contain unreleased features due for the next (major and/or minor) release of BabelStream.
Pull Requests should be made against the `develop` branch.

## Citing


Please cite BabelStream via this reference:

Deakin T, Price J, Martineau M, McIntosh-Smith S. Evaluating attainable memory bandwidth of parallel programming models via BabelStream. International Journal of Computational Science and Engineering. Special issue. Vol. 17, No. 3, pp. 247–262. 2018. DOI: 10.1504/IJCSE.2018.095847

### Other BabelStream publications

* Deakin T, Price J, Martineau M, McIntosh-Smith S. GPU-STREAM v2.0: Benchmarking the achievable memory bandwidth of many-core processors across diverse parallel programming models. 2016. Paper presented at P^3MA Workshop at ISC High Performance, Frankfurt, Germany. DOI: 10.1007/978- 3-319-46079-6_34

* Deakin T, McIntosh-Smith S. GPU-STREAM: Benchmarking the achievable memory bandwidth of Graphics Processing Units. 2015. Poster session presented at IEEE/ACM SuperComputing, Austin, United States.
  You can view the [Poster and Extended Abstract](http://sc15.supercomputing.org/sites/all/themes/SC15images/tech_poster/tech_poster_pages/post150.html).

* Deakin T, Price J, Martineau M, McIntosh-Smith S. GPU-STREAM: Now in 2D!. 2016. Poster session presented at IEEE/ACM SuperComputing, Salt Lake City, United States.
  You can view the [Poster and Extended Abstract](http://sc16.supercomputing.org/sc-archive/tech_poster/tech_poster_pages/post139.html).

* Raman K, Deakin T, Price J, McIntosh-Smith S. Improving achieved memory bandwidth from C++ codes on Intel Xeon Phi Processor (Knights Landing). IXPUG Spring Meeting, Cambridge, UK, 2017.

* Deakin T, Price J, McIntosh-Smith S. Portable methods for measuring cache hierarchy performance. 2017. Poster sessions presented at IEEE/ACM SuperComputing, Denver, United States.
  You can view the [Poster and Extended Abstract](http://sc17.supercomputing.org/SC17%20Archive/tech_poster/tech_poster_pages/post155.html)


[1]: McCalpin, John D., 1995: "Memory Bandwidth and Machine Balance in Current High Performance Computers", IEEE Computer Society Technical Committee on Computer Architecture (TCCA) Newsletter, December 1995.
