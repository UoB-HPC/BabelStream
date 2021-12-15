java-stream
===========

This is an implementation of BabelStream in Java 8 which contains the following implementations:

* `jdk-plain` - Single threaded `for`
* `jdk-stream` - Threaded implementation using JDK8's parallel stream API
* `tornadovm` - A [TornadoVM](https://github.com/beehive-lab/TornadoVM) implementation for
  PTX/OpenCL
* `aparapi` - A [Aparapi](https://git.qoto.org/aparapi/aparapi) implementation for OpenCL

### Build & Run

Prerequisites

* JDK >= 8

To run the benchmark, first create a binary:

```shell
> cd java-stream
> ./mvnw clean package
```

The binary will be located at `./target/java-stream.jar`. Run it with:

```shell
> java -version                                                                                                    ✔  11.0.11+9 ☕  tom@soraws-uk  05:03:20 
openjdk version "11.0.11" 2021-04-20
OpenJDK Runtime Environment GraalVM CE 21.1.0 (build 11.0.11+8-jvmci-21.1-b05)
OpenJDK 64-Bit Server VM GraalVM CE 21.1.0 (build 11.0.11+8-jvmci-21.1-b05, mixed mode)
> java -jar target/java-stream.jar --help
```

For best results, benchmark with the following JVM flags:

```
-XX:-UseOnStackReplacement     # disable OSR, not useful for this benchmark as we are measuring peak performance  
-XX:-TieredCompilation         # disable C1, go straight to C2 
-XX:ReservedCodeCacheSize=512m # don't flush compiled code out of cache at any point 
```

Worked example:

```shell
> java -XX:-UseOnStackReplacement -XX:-TieredCompilation -XX:ReservedCodeCacheSize=512m -jar target/java-stream.jar
BabelStream
Version: 3.4
Implementation: jdk-stream; (Java 11.0.11;Red Hat, Inc.; home=/usr/lib/jvm/java-11-openjdk-11.0.11.0.9-4.fc33.x86_64)
Running all 100 times
Precision: double
Array size: 268.4 MB (=0.3 GB)
Total size: 805.3 MB (=0.8 GB)
Function    MBytes/sec  Min (sec)   Max         Average     
Copy        17145.538   0.03131     0.04779     0.03413     
Mul         16759.092   0.03203     0.04752     0.03579     
Add         19431.954   0.04144     0.05866     0.04503     
Triad       19763.970   0.04075     0.05388     0.04510     
Dot         26646.894   0.02015     0.03013     0.02259 
```

If your OpenCL/CUDA installation is not at the default location, TornadoVM and Aparapi may fail to
detect your devices. In those cases, you may specify the library directly, for example:

```shell
> LD_PRELOAD=/opt/rocm-4.0.0/opencl/lib/libOpenCL.so.1.2 java -jar target/java-stream.jar ...
```

### Instructions for TornadoVM

The TornadoVM implementation requires you to run the binary with a patched JVM. Follow the
official [instructions](https://github.com/beehive-lab/TornadoVM/blob/master/assembly/src/docs/10_INSTALL_WITH_GRAALVM.md)
or use the following simplified instructions:

Prerequisites

* CMake >= 3.6
* GCC or clang/LLVM (GCC >= 5.5)
* Python >= 2.7
* Maven >= 3.6.3
* OpenCL headers >= 1.2 and/or CUDA SDK >= 9.0

First, get a copy of the TornadoVM source:

```shell
> cd
> git clone https://github.com/beehive-lab/TornadoVM tornadovm
```

Take note of the required GraalVM version
in `tornadovm/assembly/src/docs/10_INSTALL_WITH_GRAALVM.md`. We'll use `21.1.0` in this example.
Now, obtain a copy of GraalVM and make sure the version matches the one required by TornadoVM:

```shell
> wget https://github.com/graalvm/graalvm-ce-builds/releases/download/vm-21.1.0/graalvm-ce-java11-linux-amd64-21.1.0.tar.gz
> tar -xf graalvm-ce-java11-linux-amd64-21.1.0.tar.gz
```

Next, create `~/tornadovm/etc/sources.env` and populate the file with the following:

```shell
#!/bin/bash
export JAVA_HOME=<path to GraalVM 21.1.0 jdk>
export PATH=$PWD/bin/bin:$PATH
export TORNADO_SDK=$PWD/bin/sdk
export CMAKE_ROOT=/usr          # path to CMake binary
```

Proceed to compile TornadoVM:

```shell
> cd ~/tornadovm
> . etc/sources.env
> make graal-jdk-11-plus BACKEND={ptx,opencl}
```

To test your build, source the environment file:

```shell
> source ~/tornadovm/etc/sources.env
> LD_PRELOAD=/opt/rocm-4.0.0/opencl/lib/libOpenCL.so.1.2 tornado --devices
Number of Tornado drivers: 1
Total number of OpenCL devices  : 3
Tornado device=0:0
        AMD Accelerated Parallel Processing -- gfx1012
                Global Memory Size: 4.0 GB
                Local Memory Size: 64.0 KB
                Workgroup Dimensions: 3
                Max WorkGroup Configuration: [1024, 1024, 1024]
                Device OpenCL C version: OpenCL C 2.0

Tornado device=0:1
        Portable Computing Language -- pthread-AMD Ryzen 9 3900X 12-Core Processor
                Global Memory Size: 60.7 GB
                Local Memory Size: 8.0 MB
                Workgroup Dimensions: 3
                Max WorkGroup Configuration: [4096, 4096, 4096]
                Device OpenCL C version: OpenCL C 1.2 pocl

Tornado device=0:2
        NVIDIA CUDA -- NVIDIA GeForce GT 710
                Global Memory Size: 981.3 MB
                Local Memory Size: 48.0 KB
                Workgroup Dimensions: 3
                Max WorkGroup Configuration: [1024, 1024, 64]
                Device OpenCL C version: OpenCL C 1.2
```

You can now use TornadoVM to run java-stream:

```shell
> tornado -jar ~/java-stream/target/java-stream.jar --impl tornadovm --arraysize 65536                              1 ✘  11.0.11+9 ☕  tom@soraws-uk  05:31:34 
BabelStream
Version: 3.4
Implementation: tornadovm; (Java 11.0.11;GraalVM Community; home=~/graalvm-ce-java11-21.1.0)
Running all 100 times
Precision: double
Array size: 0.5 MB (=0.0 GB)
Total size: 1.6 MB (=0.0 GB)
Using TornadoVM device:
 - Name     : NVIDIA GeForce GT 710 CL_DEVICE_TYPE_GPU (available)
 - Id       : opencl-0-0
 - Platform : NVIDIA CUDA
 - Backend  : OpenCL
Function    MBytes/sec  Min (sec)   Max         Average     
Copy        8791.100    0.00012     0.00079     0.00015     
Mul         8774.107    0.00012     0.00061     0.00014     
Add         9903.313    0.00016     0.00030     0.00018     
Triad       9861.031    0.00016     0.00030     0.00018     
Dot         2799.465    0.00037     0.00056     0.00041
```

