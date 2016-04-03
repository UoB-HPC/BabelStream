GPU-STREAM
==========

Measure memory transfer rates to/from global device memory on GPUs.
This benchmark is similar in spirit, and based on, the STREAM benchmark [1] for CPUs.

Unlike other GPU memory bandwidth benchmarks this does *not* include the PCIe transfer time.

Usage
-----

Build the OpenCL and CUDA binaries with `make` (CUDA version requires CUDA >= v6.5)

Run the OpenCL version with `./gpu-stream-ocl` and the CUDA version with `./gpu-stream-cuda`

Automatic variation of array size
---------------------------------

I added a bash script that automatically re-runs `./gpu-stream-ocl` with different
array size and prints out results in columns, useful for plotting figures. 

    ./run-ocl.sh

    # Benchmark GPU-STREAM running on  Tesla C2070
    # Precision: double. Range: [204800 .. 102400000] step 409600
    # For more details see https://github.com/UoB-HPC/GPU-STREAM
    #  ArrayElements    ArraySize(MB)      Copy(MBytes/s)    Mul(MBytes/s)      Add(MBytes/s)      Triad(MBytes/s)
        204800            1.56             80753.117          79927.800          83088.782          83981.752          
        614400            4.68             93275.517          93897.395          94380.296          93960.518          
        1024000            7.81             98982.027          98607.910          97231.728          97690.504          
        1433600            10.93             99431.697          100773.675          99081.649          98843.119          
        1843200            14.06             100561.265          101067.177          99730.141          99793.135          
        ...         
        4300800            32.81             101984.021          101605.451          100626.557          100612.825          
        4710400            35.93             102605.629          103710.187          100675.208          100587.601          
        5120000            39.06             101817.603          102083.414          101225.864          101156.117 


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

Citing
------

You can view the [Poster and Extended Abstract](http://sc15.supercomputing.org/sites/all/themes/SC15images/tech_poster/tech_poster_pages/post150.html) on GPU-STREAM presented at SC'15. Please cite GPU-STREAM via this reference:

> Deakin T, McIntosh-Smith S. GPU-STREAM: Benchmarking the achievable memory bandwidth of Graphics Processing Units. 2015. Poster session presented at IEEE/ACM SuperComputing, Austin, United States.



[1]: McCalpin, John D., 1995: "Memory Bandwidth and Machine Balance in Current High Performance Computers", IEEE Computer Society Technical Committee on Computer Architecture (TCCA) Newsletter, December 1995.
