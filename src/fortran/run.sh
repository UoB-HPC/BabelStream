#!/bin/bash

cat ./run.sh

if [ `uname -s` == Darwin ] ; then
    NUM_HWTHREADS=`sysctl -n hw.ncpu`
    MEMORY_BYTES=`sysctl -n hw.memsize`
else
    NUM_HWTHREADS=`nproc`
    MEMORY_KILOS=`grep MemTotal /proc/meminfo | awk '{print $2}'`
fi

M=128

export OMP_NUM_THREADS=8
export OMP_PROC_BIND=close
export OMP_PLACES=threads

export ACC_NUM_CORES=${OMP_NUM_THREADS}

AFFCONTROL="numactl -N 0 -m 0 -C `seq -s "," 0 $((${OMP_NUM_THREADS}-1))`"

for compiler in gcc nvhpc cray oneapi arm amd fj ; do
    #if [ "x$compiler" == "xgcc" ] ; then
    #   export LD_PRELOAD=/usr/lib/gcc/aarch64-linux-gnu/11/libgomp.so
    #fi
    for implementation in OpenMP OpenMPTaskloop OpenMPWorkshare DoConcurrent Array OpenACC OpenACCArray CUDA CUDAKernel ; do
        if [ -f BabelStream.${compiler}.${implementation} ] ; then
            echo "BabelStream.${compiler}.${implementation}"
            ldd BabelStream.${compiler}.${implementation}
            time $AFFCONTROL \
            ./BabelStream.${compiler}.${implementation} -s $((1024*1024*${M}))
        fi
    done
done
