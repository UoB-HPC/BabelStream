#!/bin/bash

# uncomment to disable GPU targets
#HAS_GPU=0

# Orin
#if [ "x${compiler}" == "xgcc" ] ; then
#    export MCPU=cortex-a78ae
#fi
#if [ "x${compiler}" == "xarm" ] ; then
#    export MCPU=cortex-a78
#fi

COMPILERS="gcc"
if [ $(which nvfortran) ] ; then
    COMPILERS="${COMPILERS} nvhpc"
fi
if [ $(which crayftn) ] ; then
    COMPILERS="${COMPILERS} cray"
fi
if [ $(uname -m) == "aarch64" ] ; then
    if [ $(which armflang) ] ; then
        COMPILERS="${COMPILERS} arm"
    fi
    if [ $(which frt) ] ; then
        COMPILERS="${COMPILERS} fj"
    fi
elif [ $(uname -m) == "x86_64" ] ; then
    if [ $(which lscpu >& /dev/null && lscpu | grep GenuineIntel | awk '{print $3}') == "GenuineIntel" ] ; then
        COMPILERS="${COMPILERS} oneapi"
        if [ -f /opt/intel/oneapi/setvars.sh ] ; then
            . /opt/intel/oneapi/setvars.sh >& /dev/null
        fi
    else
        # ^ this detection can be improved
        COMPILERS="${COMPILERS} amd"
    fi
fi

for compiler in ${COMPILERS} ; do
    TARGETS="DoConcurrent Array OpenMP OpenMPTaskloop OpenMPWorkshare"
    if [ "${HAS_GPU}" != "0" ] ; then
        TARGETS="${TARGETS} OpenMPTarget OpenMPTargetLoop"
        if [ "x${compiler}" == "xnvhpc" ] ; then
            TARGETS="${TARGETS} CUDA CUDAKernel"
        fi
    fi
    if [ "x${compiler}" == "xnvhpc" ] || [ "x${compiler}" == "xgcc" ] || [ "x${compiler}" == "xcray" ] ; then
        TARGETS="${TARGETS} OpenACC OpenACCArray"
    fi
    for implementation in ${TARGETS} ; do
        make COMPILER=${compiler} IMPLEMENTATION=${implementation}
    done
done
