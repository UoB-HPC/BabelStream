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

CMake 3.2 or above is required.
Drivers, compiler and software applicable to whichever implementation you would like to build against. Our build system is designed to only build implementations in programming models that your system supports.

Generate the Makefile with `cmake .`

Android (outdated instructions)
------------------

Assuming you have a recent Android NDK available, you can use the
toolchain that it provides to build GPU-STREAM. You should first
use the NDK to generate a standalone toolchain:

    # Select a directory to install the toolchain to
    ANDROID_NATIVE_TOOLCHAIN=/path/to/toolchain

    ${NDK}/build/tools/make-standalone-toolchain.sh \
      --platform=android-14 \
      --toolchain=arm-linux-androideabi-4.8 \
      --install-dir=${ANDROID_NATIVE_TOOLCHAIN}

Make sure that the OpenCL headers and library (libOpenCL.so) are
available in `${ANDROID_NATIVE_TOOLCHAIN}/sysroot/usr/`.

You should then be able to build GPU-STREAM:

    make CXX=${ANDROID_NATIVE_TOOLCHAIN}/bin/arm-linux-androideabi-g++

Copy the executable and OpenCL kernels to the device:

    adb push gpu-stream-ocl /data/local/tmp
    adb push ocl-stream-kernels.cl /data/local/tmp

Run GPU-STREAM from an adb shell:

    adb shell
    cd /data/local/tmp

    # Use float if device doesn't support double, and reduce array size
    ./gpu-stream-ocl --float -n 6 -s 10000000


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
