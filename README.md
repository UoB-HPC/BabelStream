GPU-STREAM
==========

Measure memory transfer rates to/from global device memory on GPUs.
This benchmark is similar in spirit, and based on, the STREAM benchmark [1] for CPUs.

Unlike other GPU memory bandwidth benchmarks this does *not* include the PCIe transfer time.

There are multiple implementations of this benchmark in a variety of programming models.
Currently implemented are:
  - OpenCL
  - CUDA
  - OpenACC
  - OpenMP 3 and 4.5
  - Kokkos
  - RAJA
  - SYCL

Website
-------
[uob-hpc.github.io/GPU-STREAM/](https://uob-hpc.github.io/GPU-STREAM/)

Usage
-----

Drivers, compiler and software applicable to whichever implementation you would like to build against is required.

We have supplied a series of Makefiles, one for each programming model, to assist with building.
The Makefiles contain common build options, and should be simple to customise for your needs too.

General usage is `make -f <Model>.make`
Common compiler flags and names can be set by passing a `COMPILER` option to Make, e.g. `make COMPILER=GNU`.
Some models allow specifying a CPU or GPU style target, and this can be set by passing a `TARGET` option to Make, e.g. `make TARGET=GPU`.

Pass in extra flags via the `EXTRA_FLAGS` option.

The binaries are named in the form `<model>-stream`.


Results
-------

Sample results can be found in the `results` subdirectory. If you would like to submit updated results, please submit a Pull Request.

Citing
------

You can view the [Poster and Extended Abstract](http://sc15.supercomputing.org/sites/all/themes/SC15images/tech_poster/tech_poster_pages/post150.html) on GPU-STREAM presented at SC'15. Please cite GPU-STREAM via this reference:

> Deakin T, Price J, Martineau M, McIntosh-Smith S. GPU-STREAM v2.0: Benchmarking the achievable memory bandwidth of many-core processors across diverse parallel programming models. 2016. Paper presented at P^3MA Workshop at ISC High Performance, Frankfurt, Germany.

**Other GPU-STREAM publications:**

> Deakin T, McIntosh-Smith S. GPU-STREAM: Benchmarking the achievable memory bandwidth of Graphics Processing Units. 2015. Poster session presented at IEEE/ACM SuperComputing, Austin, United States.



[1]: McCalpin, John D., 1995: "Memory Bandwidth and Machine Balance in Current High Performance Computers", IEEE Computer Society Technical Committee on Computer Architecture (TCCA) Newsletter, December 1995.
