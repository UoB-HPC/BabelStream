#!/usr/bin/env bash

set -eu

# prevent ccache from caching anything for system compilers
export CCACHE_DISABLE=1

BUILD_DIR=${1:-build}
COMPILER=${2:-all}
MODEL=${3:-all}
CMAKE_BIN=${4}

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
  local install_dir="$build/install"

  # shellcheck disable=SC2086
  "$CMAKE_BIN" -B"$build" -H. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_VERBOSE_MAKEFILE=ON \
    -DCMAKE_INSTALL_PREFIX="$install_dir" \
    -DMODEL="$model" $flags &>>"$log"
  local model_lower=$(echo "$model" | awk '{print tolower($0)}')

  local cmake_code=$?

  "$CMAKE_BIN" --build "$build" -j "$(nproc)" &>>"$log"
  "$CMAKE_BIN" --build "$build" --target install -j "$(nproc)" &>>"$log"
  local cmake_code=$?
  set -e

  local bin="./$build/$model_lower-stream"
  local installed_bin="./$install_dir/bin/$model_lower-stream"

  echo "Checking for final executable: $bin"
  if [[ -f "$bin" ]]; then
    echo "$(tput setaf 2)[PASS!]($model->$build)$(tput sgr0): -DMODEL=$model $flags"
    # shellcheck disable=SC2002
    cat "$log" | sed '/^--/d' | grep -i "/bin/nvcc" | sed 's/^/    /'
    cat "$log" | sed '/^--/d' | grep -i "$grep_kw" | sed 's/^/    /'
    cat "$log" | sed '/^--/d' | grep -i "warning" | sed "s/.*/    $(tput setaf 3)&$(tput sgr0)/"
    if [[ ! -f "$installed_bin" ]]; then
      echo "$(tput setaf 1)[ERR!] looking for $installed_bin from --target install but it's not there!$(tput sgr0)"
      cat "$log"
      exit 1
    fi
  else
    echo "$(tput setaf 1)[FAIL!]($model->$build)$(tput sgr0): -DMODEL=$model $flags"
    echo "      $(tput setaf 1)CMake exited with code $cmake_code, see full build log at $log, reproduced below:$(tput sgr0)"
    cat "$log"
    exit 1
  fi
  echo "    $(tput setaf 4)$(file "$bin")$(tput sgr0)"
}

###
# KOKKOS_SRC="/home/tom/Downloads/kokkos-3.3.00"
# RAJA_SRC="/home/tom/Downloads/RAJA-v0.13.0"
#
# GCC_CXX="$(which g++-10)"
# CLANG_CXX="/usr/bin/clang++"
#
# NVHPC_SDK_DIR="/home/tom/Downloads/nvhpc_2021_219_Linux_x86_64_cuda_multi/install_components/Linux_x86_64/21.9/"
# NVHPC_NVCXX="$NVHPC_SDK_DIR/compilers/bin/nvc++"
# NVHPC_NVCC="$NVHPC_SDK_DIR/cuda/11.4/bin/nvcc"
# NVHPC_CUDA_DIR="$NVHPC_SDK_DIR/cuda/11.4"
# "$NVHPC_SDK_DIR/compilers/bin/makelocalrc" "$NVHPC_SDK_DIR/compilers/bin/" -x
#
# AOCC_CXX="/opt/AMD/aocc-compiler-2.3.0/bin/clang++"
# AOMP_CXX="/usr/lib/aomp/bin/clang++"
# OCL_LIB="/home/tom/Downloads/oclcpuexp-2020.11.11.0.04_rel/x64/libOpenCL.so"
#
# # AMD needs this rocm_path thing exported...
# export ROCM_PATH="/opt/rocm-4.5.0"
# HIP_CXX="/opt/rocm-4.5.0/bin/hipcc"
# COMPUTECPP_DIR="/home/tom/Downloads/ComputeCpp-CE-2.7.0-x86_64-linux-gnu/"
# DPCPP_DIR="/home/tom/Downloads/dpcpp_compiler"
# HIPSYCL_DIR="/opt/hipsycl/cff515c/"
#
# ICPX_CXX="/opt/intel/oneapi/compiler/2021.4.0/linux/bin/icpx"
# ICPC_CXX="/opt/intel/oneapi/compiler/2021.4.0/linux/bin/intel64/icpc"# TBB_LIB="/home/tom/Downloads/oneapi-tbb-2021.1.1/"
#
# GCC_STD_PAR_LIB="tbb"
# CLANG_STD_PAR_LIB="tbb"
# GCC_OMP_OFFLOAD_AMD=false
# GCC_OMP_OFFLOAD_NVIDIA=false
# CLANG_OMP_OFFLOAD_AMD=false
# CLANG_OMP_OFFLOAD_NVIDIA=false
###

NV_ARCH_CC="70"
AMD_ARCH="gfx_903"
NV_ARCH="sm_${NV_ARCH_CC}"
NV_ARCH_CCXY="cuda${NVHPC_CUDA_VER:?},cc80"

check_cmake_ver(){
  local current=$("$CMAKE_BIN" --version | head -n 1 | cut -d ' ' -f3)
  local required=$1
  if [ "$(printf '%s\n' "$required" "$current" | sort -V | head -n1)" = "$required" ]; then
    return 0
  else
    return 1
  fi
}

build_gcc() {
  local name="gcc_build"
  local cxx="-DCMAKE_CXX_COMPILER=${GCC_CXX:?}"

  run_build $name "${GCC_CXX:?}" omp "$cxx"
  if [ "$MODEL" = "all" ] || [ "$MODEL" = "OMP" ]; then
    # sanity check that it at least runs
    echo "Sanity checking GCC omp build..."
    "./$BUILD_DIR/omp_$name/omp-stream" -s 1048576 -n 10
  fi

  for use_onedpl in OFF OPENMP TBB; do
    case "$use_onedpl" in
      OFF) dpl_conditional_flags="-DCXX_EXTRA_LIBRARIES=${GCC_STD_PAR_LIB:-}"  ;;
      *)   dpl_conditional_flags="-DFETCH_ONEDPL=ON -DFETCH_TBB=ON -DUSE_TBB=ON -DCXX_EXTRA_FLAGS=-D_GLIBCXX_USE_TBB_PAR_BACKEND=0" ;;
    esac
    # some distributions like Ubuntu bionic implements std par with TBB, so conditionally link it here
    run_build $name "${GCC_CXX:?}" std-data    "$cxx $dpl_conditional_flags -DUSE_ONEDPL=$use_onedpl"
    run_build $name "${GCC_CXX:?}" std-indices "$cxx $dpl_conditional_flags -DUSE_ONEDPL=$use_onedpl"
    run_build $name "${GCC_CXX:?}" std-ranges  "$cxx $dpl_conditional_flags -DUSE_ONEDPL=$use_onedpl"
  done

  run_build $name "${GCC_CXX:?}" tbb "$cxx -DONE_TBB_DIR=$TBB_LIB"
  run_build $name "${GCC_CXX:?}" tbb "$cxx" # build TBB again with the system TBB
  run_build $name "${GCC_CXX:?}" tbb "$cxx -DUSE_VECTOR=ON" # build with vectors

  if [ "${GCC_OMP_OFFLOAD_AMD:-false}" != "false" ]; then
    run_build "amd_$name" "${GCC_CXX:?}" acc "$cxx -DCXX_EXTRA_FLAGS=-foffload=amdgcn-amdhsa;-fno-stack-protector;-fcf-protection=none"
    run_build "amd_$name" "${GCC_CXX:?}" omp "$cxx -DOFFLOAD=AMD:$AMD_ARCH"
  fi

  if [ "${GCC_OMP_OFFLOAD_NVIDIA:-false}" != "false" ]; then
    run_build "nvidia_$name" "${GCC_CXX:?}" acc "$cxx -DCXX_EXTRA_FLAGS=-foffload=nvptx-none;-fno-stack-protector;-fcf-protection=none"
    run_build "nvidia_$name" "${GCC_CXX:?}" omp "$cxx -DOFFLOAD=NVIDIA:$NV_ARCH"
  fi

  run_build $name "${GCC_CXX:?}" cuda "$cxx -DCMAKE_CUDA_COMPILER=${NVHPC_NVCC:?} -DCUDA_ARCH=$NV_ARCH"
  run_build $name "${GCC_CXX:?}" cuda "$cxx -DCMAKE_CUDA_COMPILER=${NVHPC_NVCC:?} -DCUDA_ARCH=$NV_ARCH -DMEM=MANAGED"
  run_build $name "${GCC_CXX:?}" cuda "$cxx -DCMAKE_CUDA_COMPILER=${NVHPC_NVCC:?} -DCUDA_ARCH=$NV_ARCH -DMEM=PAGEFAULT"
  if check_cmake_ver "3.16.0"; then
    #  run_build $name "${CC_CXX:?}" kokkos "$cxx -DKOKKOS_IN_TREE=${KOKKOS_SRC:?} -DKokkos_ENABLE_CUDA=ON"
    run_build "cuda_$name" "${GCC_CXX:?}" kokkos "$cxx -DKOKKOS_IN_TREE=${KOKKOS_SRC:?} -DKokkos_ENABLE_OPENMP=ON"
  else
    echo "Skipping Kokkos models due to CMake version requirement"
  fi
  run_build $name "${GCC_CXX:?}" ocl "$cxx -DOpenCL_LIBRARY=${OCL_LIB:?}"
  if check_cmake_ver "3.20.0"; then
    run_build $name "${GCC_CXX:?}" raja "$cxx -DRAJA_IN_TREE=${RAJA_SRC:?} -DENABLE_OPENMP=ON"
  else
    echo "Skipping RAJA models due to CMake version requirement"
  fi

  if check_cmake_ver "3.20.0"; then
   run_build "cuda_$name" "${GCC_CXX:?}" raja "$cxx -DRAJA_IN_TREE=${RAJA_SRC:?} \
     -DENABLE_CUDA=ON \
     -DTARGET=NVIDIA \
     -DCUDA_TOOLKIT_ROOT_DIR=${NVHPC_CUDA_DIR:?} \
     -DCUDA_ARCH=$NV_ARCH"
  else
    echo "Skipping RAJA models due to CMake version requirement"
  fi

  if check_cmake_ver "3.18.0"; then # CMake >= 3.15 only due to Nvidia's Thrust CMake requirements
    run_build $name "${GCC_CXX:?}" thrust "$cxx -DCMAKE_CUDA_COMPILER=${NVHPC_NVCC:?} -DCUDA_ARCH=$NV_ARCH_CC -DSDK_DIR=$NVHPC_CUDA_DIR/lib64/cmake -DTHRUST_IMPL=CUDA -DBACKEND=CUDA"
#    run_build $name "${GCC_CXX:?}" thrust "$cxx -DCMAKE_CUDA_COMPILER=${NVHPC_NVCC:?} -DCUDA_ARCH=$NV_ARCH -DSDK_DIR=$NVHPC_CUDA_DIR/lib64/cmake -DTHRUST_IMPL=CUDA -DBACKEND=OMP" # FIXME
    run_build $name "${GCC_CXX:?}" thrust "$cxx -DCMAKE_CUDA_COMPILER=${NVHPC_NVCC:?} -DCUDA_ARCH=$NV_ARCH_CC -DSDK_DIR=$NVHPC_CUDA_DIR/lib64/cmake -DTHRUST_IMPL=CUDA -DBACKEND=CPP"

    # FIXME CUDA Thrust + TBB throws the following error:
    #    /usr/lib/gcc/x86_64-linux-gnu/9/include/avx512fintrin.h(9146): error: identifier "__builtin_ia32_rndscaless_round" is undefined
    #    /usr/lib/gcc/x86_64-linux-gnu/9/include/avx512fintrin.h(9155): error: identifier "__builtin_ia32_rndscalesd_round" is undefined
    #    /usr/lib/gcc/x86_64-linux-gnu/9/include/avx512fintrin.h(14797): error: identifier "__builtin_ia32_rndscaless_round" is undefined
    #    /usr/lib/gcc/x86_64-linux-gnu/9/include/avx512fintrin.h(14806): error: identifier "__builtin_ia32_rndscalesd_round" is undefined
    #    /usr/lib/gcc/x86_64-linux-gnu/9/include/avx512dqintrin.h(1365): error: identifier "__builtin_ia32_fpclassss" is undefined
    #    /usr/lib/gcc/x86_64-linux-gnu/9/include/avx512dqintrin.h(1372): error: identifier "__builtin_ia32_fpclasssd" is undefined

    #    run_build $name "${GCC_CXX:?}" THRUST "$cxx -DCMAKE_CUDA_COMPILER=${NVHPC_NVCC:?} -DCUDA_ARCH=$NV_ARCH -DSDK_DIR=$NVHPC_CUDA_DIR/lib64/cmake -DTHRUST_IMPL=CUDA -DBACKEND=TBB"
  else
    echo "Skipping Thrust models due to CMake version requirement"
  fi

}

build_clang() {
  local name="clang_build"
  local cxx="-DCMAKE_CXX_COMPILER=${CLANG_CXX:?}"
  run_build $name "${CLANG_CXX:?}" omp "$cxx"

  if [ "${CLANG_OMP_OFFLOAD_AMD:-false}" != "false" ]; then
    run_build "amd_$name" "${GCC_CXX:?}" omp "$cxx -DOFFLOAD=AMD:$AMD_ARCH"
  fi

  if [ "${CLANG_OMP_OFFLOAD_NVIDIA:-false}" != "false" ]; then
    run_build "nvidia_$name" "${GCC_CXX:?}" omp "$cxx -DOFFLOAD=NVIDIA:$NV_ARCH"
  fi

  if check_cmake_ver "3.20.0"; then
    run_build $name "${CLANG_CXX:?}" raja "$cxx -DRAJA_IN_TREE=${RAJA_SRC:?} -DENABLE_OPENMP=ON"
  else
    echo "Skipping RAJA models due to CMake version requirement"
  fi
  run_build $name "${CLANG_CXX:?}" cuda "$cxx -DCMAKE_CUDA_COMPILER=${NVHPC_NVCC:?} -DCUDA_ARCH=$NV_ARCH"
  run_build $name "${CLANG_CXX:?}" cuda "$cxx -DCMAKE_CUDA_COMPILER=${NVHPC_NVCC:?} -DCUDA_ARCH=$NV_ARCH -DMEM=MANAGED"
  run_build $name "${CLANG_CXX:?}" cuda "$cxx -DCMAKE_CUDA_COMPILER=${NVHPC_NVCC:?} -DCUDA_ARCH=$NV_ARCH -DMEM=PAGEFAULT"
  if check_cmake_ver "3.16.0"; then
    run_build $name "${CLANG_CXX:?}" kokkos "$cxx -DKOKKOS_IN_TREE=${KOKKOS_SRC:?} -DKokkos_ENABLE_OPENMP=ON"
  else
    echo "Skipping Kokkos models due to CMake version requirement"
  fi
  run_build $name "${CLANG_CXX:?}" ocl "$cxx -DOpenCL_LIBRARY=${OCL_LIB:?}"

  for use_onedpl in OFF OPENMP TBB; do
    case "$use_onedpl" in
      OFF) dpl_conditional_flags="-DCXX_EXTRA_LIBRARIES=${CLANG_STD_PAR_LIB:-}" ;;
      *)   dpl_conditional_flags="-DFETCH_ONEDPL=ON -DFETCH_TBB=ON -DUSE_TBB=ON -DCXX_EXTRA_FLAGS=-D_GLIBCXX_USE_TBB_PAR_BACKEND=0"  ;;
    esac
    run_build $name "${CLANG_CXX:?}" std-data     "$cxx $dpl_conditional_flags -DUSE_ONEDPL=$use_onedpl"
    run_build $name "${CLANG_CXX:?}" std-indices  "$cxx $dpl_conditional_flags -DUSE_ONEDPL=$use_onedpl"
    # run_build $name "${CLANG_CXX:?}" std-ranges "$cxx $dpl_conditional_flags -DUSE_ONEDPL=$use_onedpl" # not yet supported
  done

  run_build $name "${CLANG_CXX:?}" tbb "$cxx -DONE_TBB_DIR=$TBB_LIB"
  run_build $name "${CLANG_CXX:?}" tbb "$cxx" # build TBB again with the system TBB
  run_build $name "${CLANG_CXX:?}" tbb "$cxx -DUSE_VECTOR=ON" # build with vectors
  if check_cmake_ver "3.20.0"; then
    run_build $name "${CLANG_CXX:?}" raja "$cxx -DRAJA_IN_TREE=${RAJA_SRC:?} -DENABLE_OPENMP=ON"
  else
    echo "Skipping RAJA models due to CMake version requirement"
  fi
  # no clang /w RAJA+cuda because it needs nvcc which needs gcc
}

build_nvhpc() {
  local name="nvhpc_build"
  local cxx="-DCMAKE_CXX_COMPILER=${NVHPC_NVCXX:?}"
  run_build $name "${NVHPC_NVCXX:?}" std-data "$cxx -DNVHPC_OFFLOAD=$NV_ARCH_CCXY"
  run_build $name "${NVHPC_NVCXX:?}" std-indices "$cxx -DNVHPC_OFFLOAD=$NV_ARCH_CCXY"

  run_build $name "${NVHPC_NVCXX:?}" acc "$cxx -DTARGET_DEVICE=gpu -DTARGET_PROCESSOR=px -DCUDA_ARCH=$NV_ARCH_CCXY"
  run_build $name "${NVHPC_NVCXX:?}" acc "$cxx -DTARGET_DEVICE=multicore -DTARGET_PROCESSOR=zen"
}

build_aocc() {
  run_build aocc_build "${AOCC_CXX:?}" omp "-DCMAKE_CXX_COMPILER=${AOCC_CXX:?}"
}

build_aomp() {
  run_build aomp_amd_build "${AOMP_CXX:?}" omp "-DCMAKE_CXX_COMPILER=${AOMP_CXX:?} -DOFFLOAD=AMD:gfx906"
  #run_build aomp_nvidia_build "-DCMAKE_CXX_COMPILER=${AOMP_CXX:?} -DOFFLOAD=NVIDIA:$NV_ARCH"
}

build_hip() {
  local name="hip_build"

  run_build $name "${HIP_CXX:?}" hip "-DCMAKE_CXX_COMPILER=${HIP_CXX:?}"
  run_build $name "${HIP_CXX:?}" hip "-DCMAKE_CXX_COMPILER=${HIP_CXX:?} -DMEM=MANAGED"
  run_build $name "${HIP_CXX:?}" hip "-DCMAKE_CXX_COMPILER=${HIP_CXX:?} -DMEM=PAGEFAULT"

  run_build $name "${GCC_CXX:?}" thrust "-DCMAKE_CXX_COMPILER=${HIP_CXX:?} -DSDK_DIR=$ROCM_PATH -DTHRUST_IMPL=ROCM"
}

build_icpx() {
  # clang derived
  set +u
  source /opt/intel/oneapi/setvars.sh -force || true
  set -u
  run_build intel_build "${ICPX_CXX:?}" omp "-DCMAKE_CXX_COMPILER=${ICPX_CXX:?} -DOFFLOAD=INTEL"
}

build_icpc() {
  # icc/icpc
  set +u
  source /opt/intel/oneapi/setvars.sh -force || true
  set -u
  local name="intel_build"
  local cxx="-DCMAKE_CXX_COMPILER=${ICPC_CXX:?}"
  run_build $name "${ICPC_CXX:?}" omp "$cxx"
  run_build $name "${ICPC_CXX:?}" ocl "$cxx -DOpenCL_LIBRARY=${OCL_LIB:?}"
  if check_cmake_ver "3.20.0"; then
    run_build $name "${ICPC_CXX:?}" raja "$cxx -DRAJA_IN_TREE=${RAJA_SRC:?} -DENABLE_OPENMP=ON"
  else
    echo "Skipping RAJA models due to CMake version requirement"
  fi

  if check_cmake_ver "3.16.0"; then
    run_build $name "${ICPC_CXX:?}" kokkos "$cxx -DKOKKOS_IN_TREE=${KOKKOS_SRC:?} -DKokkos_ENABLE_OPENMP=ON"
  else
    echo "Skipping Kokkos models due to CMake version requirement"
  fi

}

build_dpcpp() {
  run_build intel_build "${DPCPP_DIR:?}" sycl "-DCMAKE_CXX_COMPILER=${GCC_CXX:?} \
  -DSYCL_COMPILER=DPCPP \
  -DSYCL_COMPILER_DIR=${DPCPP_DIR:?}"

  #  for oneAPI BaseKit:
  #  source /opt/intel/oneapi/setvars.sh -force
  #  run_build intel_build "dpcpp" sycl "-DCMAKE_CXX_COMPILER=${GCC_CXX:?} -DSYCL_COMPILER=ONEAPI-DPCPP"
}

build_hipsycl() {
  run_build hipsycl_build "syclcc" sycl "
  -DSYCL_COMPILER=HIPSYCL \
  -DSYCL_COMPILER_DIR=${HIPSYCL_DIR:?}"
}

echo "Test compiling with ${COMPILER} CXX for ${MODEL} model"
"$CMAKE_BIN" --version

case "$COMPILER" in
gcc) build_gcc ;;
clang) build_clang ;;
nvhpc) build_nvhpc ;;
aocc) build_aocc ;;
aomp) build_aomp ;;
hip) build_hip ;;
dpcpp) build_dpcpp ;;
hipsycl) build_hipsycl ;;

# XXX below are local only; licence or very large download required, candidate for local runner
computecpp) build_computecpp ;;
icpx) build_icpx ;;
icpc) build_icpc ;;

all)
  build_gcc
  build_clang
  build_nvhpc
  build_aocc
  build_aomp
  build_hip
  build_dpcpp
  build_hipsycl

  build_computecpp
  build_icpx
  build_icpc

  ;;
*)
  echo "Unknown $COMPILER, use ALL to compile with all supported compilers"
  ;;
esac
