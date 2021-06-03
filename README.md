BabelStream
==========

<img src="https://github.com/UoB-HPC/BabelStream/blob/gh-pages/img/BabelStreamlogo.png?raw=true" alt="logo" height="300" align="right" />


Measure memory transfer rates to/from global device memory on GPUs.
This benchmark is similar in spirit, and based on, the STREAM benchmark [1] for CPUs.

Unlike other GPU memory bandwidth benchmarks this does *not* include the PCIe transfer time.

There are multiple implementations of this benchmark in a variety of programming models.
Currently implemented are:
  - OpenCL
  - CUDA
  - OpenACC
  - OpenMP 3 and 4.5
  - C++ Parallel STL
  - Kokkos
  - RAJA
  - SYCL
  - TBB

This code was previously called GPU-STREAM.


How is this different to STREAM?
--------------------------------

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

> Van der Wijngaart, Rob F., and Timothy G. Mattson. The parallel research kernels. IEEE High Performance Extreme Computing Conference (HPEC). IEEE, 2014.

> R. F. Van der Wijngaart, A. Kayi, J. R. Hammond, G. Jost, T. St. John, S. Sridharan, T. G. Mattson, J. Abercrombie, and J. Nelson. Comparing runtime systems with exascale ambitions using the Parallel Research Kernels. ISC 2016, [DOI: 10.1007/978-3-319-41321-1_17](https://doi.org/10.1007/978-3-319-41321-1_17).

> Jeff R. Hammond and Timothy G. Mattson. Evaluating data parallelism in C++ using the Parallel Research Kernels. IWOCL 2019, [DOI: 10.1145/3318170.3318192](https://doi.org/10.1145/3318170.3318192).


Website
-------
[uob-hpc.github.io/BabelStream/](https://uob-hpc.github.io/BabelStream/)

Usage
-----

Drivers, compiler and software applicable to whichever implementation you would like to build against is required.

### CMake

The project supports building with CMake >= 3.13.0, it can be installed without root via the [official script](https://cmake.org/download/).
As with any CMake project, first configure the project:

```shell
> cd babelstream
> cmake -Bbuild -H. -DMODEL=<model> <model specific flags prefixed with -D...> # configure the build, build type defaults to Release 
> cmake --build build # compile it 
> ./build/babelstream # executable available at ./build/
```

By default, we have defined a set of optimal flags for known HPC compilers.
There are assigned those to `RELEASE_FLAGS`, and you can override them if required.

To find out what flag each model supports or requires, simply configure while only specifying the model.
For example:
```shell
> cd babelstream
> cmake -Bbuild -H. -DMODEL=OCL 
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
-- Available models:  OMP;OCL;STD;STD20;HIP;CUDA;KOKKOS;SYCL;ACC;RAJA;TBB
-- Selected model  :  OCL
-- Supported flags:

   CMAKE_CXX_COMPILER (optional, default=c++): Any CXX compiler that is supported by CMake detection
   OpenCL_LIBRARY (optional, default=): Path to OpenCL library, usually called libOpenCL.so
...
```
Alternatively, refer to the [CI script](./ci-test-compile.sh), which test-compiles most of the models, and see which flags are used there.

*It is recommended that you delete the `build` directory when you change any of the build flags.* 

### GNU Make

We have supplied a series of Makefiles, one for each programming model, to assist with building.
The Makefiles contain common build options, and should be simple to customise for your needs too.

General usage is `make -f <Model>.make`
Common compiler flags and names can be set by passing a `COMPILER` option to Make, e.g. `make COMPILER=GNU`.
Some models allow specifying a CPU or GPU style target, and this can be set by passing a `TARGET` option to Make, e.g. `make TARGET=GPU`.

Pass in extra flags via the `EXTRA_FLAGS` option.

The binaries are named in the form `<model>-stream`.

#### Building Kokkos for Make

Kokkos version >= 3 requires setting the `KOKKOS_PATH` flag to the *source* directory of a distribution. 
For example:

```
cd 
wget https://github.com/kokkos/kokkos/archive/3.1.01.tar.gz
tar -xvf 3.1.01.tar.gz # should end up with ~/kokkos-3.1.01
cd BabelStream
make -f Kokkos.make KOKKOS_PATH=~/kokkos-3.1.01 
```
See make output for more information on supported flags.

#### Building RAJA for Make

We use the following command to build RAJA using the Intel Compiler.
```
cmake .. -DCMAKE_INSTALL_PREFIX=<prefix> -DCMAKE_C_COMPILER=icc -DCMAKE_CXX_COMPILER=icpc -DRAJA_PTR="RAJA_USE_RESTRICT_ALIGNED_PTR" -DCMAKE_BUILD_TYPE=ICCBuild -DRAJA_ENABLE_TESTS=Off
```
For building with CUDA support, we use the following command.
```
cmake .. -DCMAKE_INSTALL_PREFIX=<prefix> -DRAJA_PTR="RAJA_USE_RESTRICT_ALIGNED_PTR" -DRAJA_ENABLE_CUDA=1 -DRAJA_ENABLE_TESTS=Off
```

Results
-------

Sample results can be found in the `results` subdirectory. If you would like to submit updated results, please submit a Pull Request.

Contributing
------------

As of v4.0, the `main` branch of this repository will hold the latest released version.

The `develop` branch will contain unreleased features due for the next (major and/or minor) release of BabelStream.
Pull Requests should be made against the `develop` branch.

Citing
------

Please cite BabelStream via this reference:

> Deakin T, Price J, Martineau M, McIntosh-Smith S. GPU-STREAM v2.0: Benchmarking the achievable memory bandwidth of many-core processors across diverse parallel programming models. 2016. Paper presented at P^3MA Workshop at ISC High Performance, Frankfurt, Germany.

**Other BabelStream publications:**

> Deakin T, McIntosh-Smith S. GPU-STREAM: Benchmarking the achievable memory bandwidth of Graphics Processing Units. 2015. Poster session presented at IEEE/ACM SuperComputing, Austin, United States.
You can view the [Poster and Extended Abstract](http://sc15.supercomputing.org/sites/all/themes/SC15images/tech_poster/tech_poster_pages/post150.html).

> Deakin T, Price J, Martineau M, McIntosh-Smith S. GPU-STREAM: Now in 2D!. 2016. Poster session presented at IEEE/ACM SuperComputing, Salt Lake City, United States.
You can view the [Poster and Extended Abstract](http://sc16.supercomputing.org/sc-archive/tech_poster/tech_poster_pages/post139.html).

> Raman K, Deakin T, Price J, McIntosh-Smith S. Improving achieved memory bandwidth from C++ codes on Intel Xeon Phi Processor (Knights Landing). IXPUG Spring Meeting, Cambridge, UK, 2017.

> Deakin T, Price J, Martineau M, McIntosh-Smith S. Evaluating attainable memory bandwidth of parallel programming models via BabelStream. International Journal of Computational Science and Engineering. Special issue (in press). 2017.

> Deakin T, Price J, McIntosh-Smith S. Portable methods for measuring cache hierarchy performance. 2017. Poster sessions presented at IEEE/ACM SuperComputing, Denver, United States.
You can view the [Poster and Extended Abstract](http://sc17.supercomputing.org/SC17%20Archive/tech_poster/tech_poster_pages/post155.html)


[1]: McCalpin, John D., 1995: "Memory Bandwidth and Machine Balance in Current High Performance Computers", IEEE Computer Society Technical Committee on Computer Architecture (TCCA) Newsletter, December 1995.
