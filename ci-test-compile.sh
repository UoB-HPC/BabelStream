#!/usr/bin/env bash

set -eu

# prevent ccache from caching anything for system compilers
export CCACHE_DISABLE=1

BUILD_DIR=${1:-build}
COMPILER=${2:-all}
MODEL=${3:-all}

LOG_DIR="$BUILD_DIR"

mkdir -p "$LOG_DIR"

if [ "${GITHUB_ACTIONS:-false}" = true ]; then
  echo "Running in GitHub Actions, setting TERM..."
  TERM=xterm
  export TERM=xterm
fi

function_exists() {
  declare -f -F "$1" >/dev/null
  return $?
}

run_build() {
  local key="$1"
  local grep_kw="$2"
  local model="$3"
  local flags="$4"

  if [ "$MODEL" != "all" ] && [ "$MODEL" != "$model" ]; then
    echo "Skipping -DMODEL=$model $flags"
    return 0
  fi

  local log="$LOG_DIR/${model}_${key}.log"
  rm -f "$log"
  touch "$log"

  local build="$BUILD_DIR/${model}_${key}"

  rm -rf "$build"
  set +e

  # shellcheck disable=SC2086
  cmake -B"$build" -H. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_VERBOSE_MAKEFILE=ON \
    -DMODEL="$model" $flags &>>"$log"
  local cmake_code=$?

  cmake --build "$build" --target babelstream -j "$(nproc)" &>>"$log"
  local cmake_code=$?
  set -e

  local bin="./$build/babelstream"
  if [[ -f "$bin" ]]; then
    echo "$(tput setaf 2)[PASS!]($model->$build)$(tput sgr0): -DMODEL=$model $flags"
    # shellcheck disable=SC2002
    cat "$log" | sed '/^--/d' | grep -i "$grep_kw" | sed 's/^/    /'
  else
    echo "$(tput setaf 1)[FAIL!]($model->$build)$(tput sgr0): -DMODEL=$model $flags"
    echo "      CMake exited with code $cmake_code, see full build log at $log, reproduced below:"
    cat "$log"
    exit 1
  fi
  echo "    $(tput setaf 4)$(file "$bin")$(tput sgr0)"
}

##
#KOKKOS_SRC="/home/tom/Downloads/kokkos-3.3.00"
#RAJA_SRC="/home/tom/Downloads/RAJA-v0.13.0"
#
#GCC_CXX="/usr/bin/g++"
#CLANG_CXX="/usr/bin/clang++"
#
#NVSDK="/home/tom/Downloads/nvhpc_2021_212_Linux_x86_64_cuda_11.2/install_components/Linux_x86_64/21.2/"
#NVHPC_NVCXX="$NVSDK/compilers/bin/nvc++"
#NVHPC_NVCC="$NVSDK/cuda/11.2/bin/nvcc"
#NVHPC_CUDA_DIR="$NVSDK/cuda/11.2"
#"$NVSDK/compilers/bin/makelocalrc" "$NVSDK/compilers/bin/" -x
#
##NVHPC_NVCXX="/opt/nvidia/hpc_sdk/Linux_x86_64/21.1/compilers/bin/nvc++"
##NVHPC_NVCC="/opt/nvidia/hpc_sdk/Linux_x86_64/21.1/cuda/11.2/bin/nvcc"
##NVHPC_CUDA_DIR="/opt/nvidia/hpc_sdk/Linux_x86_64/21.1/cuda/11.2"
#
#AOCC_CXX="/opt/AMD/aocc-compiler-2.3.0/bin/clang++"
#AOMP_CXX="/usr/lib/aomp/bin/clang++"
#OCL_LIB="/home/tom/Downloads/oclcpuexp-2020.11.11.0.04_rel/x64/libOpenCL.so"
#
## AMD needs this rocm_path thing exported...
#export ROCM_PATH="/opt/rocm-4.0.0"
#HIP_CXX="/opt/rocm-4.0.0/bin/hipcc"
#COMPUTECPP_DIR="/home/tom/Desktop/computecpp_archive/ComputeCpp-CE-2.3.0-x86_64-linux-gnu"
#DPCPP_DIR="/home/tom/Downloads/dpcpp_compiler"
#HIPSYCL_DIR="/opt/hipsycl/cff515c/"
#
#ICPX_CXX="/opt/intel/oneapi/compiler/2021.1.2/linux/bin/icpx"
#
#GCC_STD_PAR_LIB=""
#CLANG_STD_PAR_LIB=""
#GCC_OMP_OFFLOAD_AMD=false
#GCC_OMP_OFFLOAD_NVIDIA=true
#CLANG_OMP_OFFLOAD_AMD=false
#CLANG_OMP_OFFLOAD_NVIDIA=false

AMD_ARCH="gfx_903"
NV_ARCH="sm_70"
NV_ARCH_CCXY="cuda11.2,cc80"

build_gcc() {
  local name="gcc_build"
  local cxx="-DCMAKE_CXX_COMPILER=${GCC_CXX:?}"

  run_build $name "${GCC_CXX:?}" OMP "$cxx"
  if [ "$MODEL" = "all" ] || [ "$MODEL" = "OMP" ]; then
    # sanity check that it at least runs
    echo "Sanity checking GCC OMP build..."
    "./$BUILD_DIR/OMP_$name/babelstream" -s 1048576 -n 10
  fi

  # some distributions like Ubuntu bionic implements std par with TBB, so conditionally link it here
  run_build $name "${GCC_CXX:?}" STD "$cxx -DCXX_EXTRA_LIBRARIES=${GCC_STD_PAR_LIB:-}"
  run_build $name "${GCC_CXX:?}" STD20 "$cxx -DCXX_EXTRA_LIBRARIES=${GCC_STD_PAR_LIB:-}"

  if [ "${GCC_OMP_OFFLOAD_AMD:-false}" != "false" ]; then
    run_build "amd_$name" "${GCC_CXX:?}" ACC "$cxx -DCXX_EXTRA_FLAGS=-foffload=amdgcn-amdhsa"
    run_build "amd_$name" "${GCC_CXX:?}" OMP "$cxx -DOFFLOAD=AMD:$AMD_ARCH"
  fi

  if [ "${GCC_OMP_OFFLOAD_NVIDIA:-false}" != "false" ]; then
    run_build "nvidia_$name" "${GCC_CXX:?}" ACC "$cxx -DCXX_EXTRA_FLAGS=-foffload=nvptx-none"
    run_build "nvidia_$name" "${GCC_CXX:?}" OMP "$cxx -DOFFLOAD=NVIDIA:$NV_ARCH"
  fi

  run_build $name "${GCC_CXX:?}" CUDA "$cxx -DCMAKE_CUDA_COMPILER=${NVHPC_NVCC:?} -DCUDA_ARCH=$NV_ARCH"
  run_build $name "${GCC_CXX:?}" CUDA "$cxx -DCMAKE_CUDA_COMPILER=${NVHPC_NVCC:?} -DCUDA_ARCH=$NV_ARCH -DMEM=MANAGED"
  run_build $name "${GCC_CXX:?}" CUDA "$cxx -DCMAKE_CUDA_COMPILER=${NVHPC_NVCC:?} -DCUDA_ARCH=$NV_ARCH -DMEM=PAGEFAULT"
  #  run_build $name "${CC_CXX:?}" KOKKOS "$cxx -DKOKKOS_IN_TREE=${KOKKOS_SRC:?} -DKokkos_ENABLE_CUDA=ON"
  run_build "cuda_$name" "${GCC_CXX:?}" KOKKOS "$cxx -DKOKKOS_IN_TREE=${KOKKOS_SRC:?} -DKokkos_ENABLE_OPENMP=ON"
  run_build $name "${GCC_CXX:?}" OCL "$cxx -DOpenCL_LIBRARY=${OCL_LIB:?}"
  run_build $name "${GCC_CXX:?}" RAJA "$cxx -DRAJA_IN_TREE=${RAJA_SRC:?}"
  run_build "cuda_$name" "${GCC_CXX:?}" RAJA "$cxx -DRAJA_IN_TREE=${RAJA_SRC:?} \
  -DENABLE_CUDA=ON \
  -DTARGET=NVIDIA \
  -DCUDA_TOOLKIT_ROOT_DIR=${NVHPC_CUDA_DIR:?} \
  -DCUDA_ARCH=$NV_ARCH"

}

build_clang() {
  local name="clang_build"
  local cxx="-DCMAKE_CXX_COMPILER=${CLANG_CXX:?}"
  run_build $name "${CLANG_CXX:?}" OMP "$cxx"

  if [ "${CLANG_OMP_OFFLOAD_AMD:-false}" != "false" ]; then
    run_build "amd_$name" "${GCC_CXX:?}" OMP "$cxx -DOFFLOAD=AMD:$AMD_ARCH"
  fi

  if [ "${CLANG_OMP_OFFLOAD_NVIDIA:-false}" != "false" ]; then
    run_build "nvidia_$name" "${GCC_CXX:?}" OMP "$cxx -DOFFLOAD=NVIDIA:$NV_ARCH"
  fi

  run_build $name "${CLANG_CXX:?}" CUDA "$cxx -DCMAKE_CUDA_COMPILER=${NVHPC_NVCC:?} -DCUDA_ARCH=$NV_ARCH"
  run_build $name "${CLANG_CXX:?}" CUDA "$cxx -DCMAKE_CUDA_COMPILER=${NVHPC_NVCC:?} -DCUDA_ARCH=$NV_ARCH -DMEM=MANAGED"
  run_build $name "${CLANG_CXX:?}" CUDA "$cxx -DCMAKE_CUDA_COMPILER=${NVHPC_NVCC:?} -DCUDA_ARCH=$NV_ARCH -DMEM=PAGEFAULT"
  run_build $name "${CLANG_CXX:?}" KOKKOS "$cxx -DKOKKOS_IN_TREE=${KOKKOS_SRC:?} -DKokkos_ENABLE_OPENMP=ON"
  run_build $name "${CLANG_CXX:?}" OCL "$cxx -DOpenCL_LIBRARY=${OCL_LIB:?}"
  run_build $name "${CLANG_CXX:?}" STD "$cxx -DCXX_EXTRA_LIBRARIES=${CLANG_STD_PAR_LIB:-}"
  # run_build $name "${LANG_CXX:?}" STD20 "$cxx -DCXX_EXTRA_LIBRARIES=${CLANG_STD_PAR_LIB:-}" # not yet supported
  run_build $name "${CLANG_CXX:?}" RAJA "$cxx -DRAJA_IN_TREE=${RAJA_SRC:?}"
  # no clang /w RAJA+cuda because it needs nvcc which needs gcc
}

build_nvhpc() {
  local name="nvhpc_build"
  local cxx="-DCMAKE_CXX_COMPILER=${NVHPC_NVCXX:?}"
  run_build $name "${NVHPC_NVCXX:?}" STD "$cxx -DNVHPC_OFFLOAD=$NV_ARCH_CCXY"
  run_build $name "${NVHPC_NVCXX:?}" ACC "$cxx -DTARGET_DEVICE=gpu -DTARGET_PROCESSOR=px -DCUDA_ARCH=$NV_ARCH_CCXY"
  run_build $name "${NVHPC_NVCXX:?}" ACC "$cxx -DTARGET_DEVICE=multicore -DTARGET_PROCESSOR=zen"
}

build_aocc() {
  run_build aocc_build "${AOCC_CXX:?}" OMP "-DCMAKE_CXX_COMPILER=${AOCC_CXX:?}"
}

build_aomp() {
  run_build aomp_amd_build "${AOMP_CXX:?}" OMP "-DCMAKE_CXX_COMPILER=${AOMP_CXX:?} -DOFFLOAD=AMD:gfx906"
  #run_build aomp_nvidia_build "-DCMAKE_CXX_COMPILER=${AOMP_CXX:?} -DOFFLOAD=NVIDIA:$NV_ARCH"
}

build_hip() {
  run_build hip_build "${HIP_CXX:?}" HIP "-DCMAKE_CXX_COMPILER=${HIP_CXX:?}"
}

build_icpx() {
  source /opt/intel/oneapi/setvars.sh -force
  run_build intel_build "${ICPX_CXX:?}" OMP "-DCMAKE_CXX_COMPILER=${ICPX_CXX:?} -DOFFLOAD=INTEL"
}

build_computecpp() {
  run_build computecpp_build "compute++" SYCL "-DCMAKE_CXX_COMPILER=${GCC_CXX:?} \
  -DSYCL_COMPILER=COMPUTECPP \
  -DSYCL_COMPILER_DIR=${COMPUTECPP_DIR:?} \
  -DOpenCL_LIBRARY=${OCL_LIB:?}"
}

build_dpcpp() {
  run_build intel_build "${DPCPP_DIR:?}" SYCL "-DCMAKE_CXX_COMPILER=${GCC_CXX:?} \
  -DSYCL_COMPILER=DPCPP \
  -DSYCL_COMPILER_DIR=${DPCPP_DIR:?}"

  #  for oneAPI BaseKit:
  #  source /opt/intel/oneapi/setvars.sh -force
  #  run_build intel_build "dpcpp" SYCL "-DCMAKE_CXX_COMPILER=${GCC_CXX:?} -DSYCL_COMPILER=ONEAPI-DPCPP"
}

build_hipsycl() {
  run_build hipsycl_build "syclcc" SYCL "
  -DSYCL_COMPILER=HIPSYCL \
  -DSYCL_COMPILER_DIR=${HIPSYCL_DIR:?}"
}

# TODO tested locally but can't install compilers for these two remotely without registration/license:
# build_icpx
# build_computecpp

case "$COMPILER" in
gcc) build_gcc ;;
clang) build_clang ;;
nvhpc) build_nvhpc ;;
aocc) build_aocc ;;
aomp) build_aomp ;;
hip) build_hip ;;
dpcpp) build_dpcpp ;;
hipsycl) build_hipsycl ;;
all)
  build_gcc
  build_clang
  build_nvhpc
  build_aocc
  build_aomp
  build_hip
  build_dpcpp
  build_hipsycl
  ;;
*)
  echo "Unknown $COMPILER, use ALL to compile with all supported compilers"
  ;;
esac
