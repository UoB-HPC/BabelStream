GPU-STREAM
==========

Measure memory transfer rates to/from global device memory on GPUs.
This benchmark is similar in spirit, and based on, the STREAM benchmark [1] for CPUs.

Unlike other GPU memory bandwidth benchmarks this does *not* include the PCIe transfer time.

Usage
-----

Build the OpenCL and CUDA binaries with `make` (CUDA version requires CUDA >= v6.5)

Run the OpenCL version with `./gpu-stream-ocl` and the CUDA version with `./gpu-stream-cuda`

Android
-------

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

[1]: McCalpin, John D., 1995: "Memory Bandwidth and Machine Balance in Current High Performance Computers", IEEE Computer Society Technical Committee on Computer Architecture (TCCA) Newsletter, December 1995.
