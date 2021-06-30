#!/usr/bin/env bash

set -eu

WORK_DIR="${1:-.}"
MODE="${2:-SETUP}"
PARALLEL="${3:-false}"

FORCE_DOWNLOAD=false

if [ "$_" = "$0" ]; then
  echo "This script must be sourced for the exports to work!"
  exit 1
fi

case "$MODE" in
SETUP)
  SETUP=true
  echo "Preparing env with setup..."
  ;;
VARS)
  SETUP=false
  echo "Preparing env without setup..."
  ;;
*)
  echo "Bad option"
  echo "$0 <work_dir:dir> <VARS|SETUP> <parallel:bool>"
  exit 1
  ;;
esac

mkdir -p "$WORK_DIR"
PREV_DIR="$PWD"
cd "$WORK_DIR" || exit 1

export_var() {
  export "$1"="$2"
  # see
  # https://docs.github.com/en/actions/reference/workflow-commands-for-github-actions#setting-an-environment-variable
  if [ "${GITHUB_ACTIONS:-false}" = true ]; then
    echo "$1=$2" >>"$GITHUB_ENV"
  fi
}

check_size() {
  if [ "$SETUP" = true ]; then
    echo "Used space for $PWD"
    du -sh .
    df -h .
  fi
}

get_and_install_deb() {

  local name="$1"
  local install_dir="$2"
  local pkg_url="$3"
  shift
  local wget_args=("$@")

  local pkg_name="$name.deb"

  if [ "$SETUP" = true ]; then
    if [ ! -f "$pkg_name" ] || [ "$FORCE_DOWNLOAD" = true ]; then
      echo "$pkg_name not found, downloading"
      rm -f "$pkg_name"
      # shellcheck disable=SC2086
      wget -q --show-progress --progress=bar:force:noscroll "${wget_args[@]}" "$pkg_url" -O "$pkg_name"
    fi
    #    rm -rf "$install_dir"
    echo "Preparing to install $pkg_name locally to $install_dir ..."
    dpkg-deb -x "$pkg_name" "$install_dir"
    echo "$pkg_name installed, deleting $pkg_name ..."
    rm -f "$pkg_name" # delete for space
  fi

}

get() {
  local name="$1"
  local pkg_url="$2"
  if [ "$SETUP" = true ]; then
    if [ ! -f "$name" ] || [ "$FORCE_DOWNLOAD" = true ]; then
      echo "$name not found, downloading..."
      wget -q --show-progress --progress=bar:force:noscroll "$pkg_url" -O "$name"
    fi
  fi
}

get_and_untar() {
  local name="$1"
  local pkg_url="$2"
  if [ "$SETUP" = true ]; then
    if [ ! -f "$name" ] || [ "$FORCE_DOWNLOAD" = true ]; then
      echo "$name not found, downloading..."
      wget -q --show-progress --progress=bar:force:noscroll "$pkg_url" -O "$name"
    fi
    echo "Preparing to extract $name ..."
    tar -xf "$name"
    echo "$name extracted, deleting archive ..."
    rm -f "$name" # delete for space
  fi
}

verify_bin_exists() {
  if [ ! -f "$1" ]; then
    echo "[FAIL] $1 does not exist or is not a file!"
    exit 1
  else echo "[OK! ] $1"; fi
}

verify_dir_exists() {
  if [ ! -d "$1" ]; then
    echo "[FAIL] $1 does not exist or is not a directory!"
    exit 1
  else echo "[OK! ] $1"; fi
}

setup_aocc() {
  echo "Preparing AOCC"

  local aocc_ver="2.3.0"
  local tarball="aocc-$aocc_ver.tar.xz"
  # XXX it's actually XZ compressed, so it should be tar.xz
  local AOCC_URL="http://developer.amd.com/wordpress/media/files/aocc-compiler-2.3.0.tar"
  # local AOCC_URL="http://localhost:8000/aocc-compiler-2.3.0.tar"

  get_and_untar "$tarball" "$AOCC_URL"
  export_var AOCC_CXX "$PWD/aocc-compiler-$aocc_ver/bin/clang++"
  verify_bin_exists "$AOCC_CXX"
  "$AOCC_CXX" --version
  check_size
}

setup_nvhpc() {
  echo "Preparing Nvidia HPC SDK"
  local tarball="nvhpc.tar.gz"
#  local url="http://localhost:8000/nvhpc_2021_215_Linux_x86_64_cuda_11.3.tar.gz"
  local url="https://developer.download.nvidia.com/hpc-sdk/21.5/nvhpc_2021_215_Linux_x86_64_cuda_11.3.tar.gz"
  get_and_untar "$tarball" "$url"

  local sdk_dir="$PWD/nvhpc_2021_215_Linux_x86_64_cuda_11.3/install_components/Linux_x86_64/21.5"
  local bin_dir="$sdk_dir/compilers/bin"
  "$bin_dir/makelocalrc" "$bin_dir" -x

  export_var NVHPC_NVCXX "$bin_dir/nvc++"
  export_var NVHPC_NVCC "$sdk_dir/cuda/11.3/bin/nvcc"
  export_var NVHPC_CUDA_DIR "$sdk_dir/cuda/11.3"
  echo "Installed CUDA versions:"
  ls "$sdk_dir/cuda"
  verify_bin_exists "$NVHPC_NVCXX"
  verify_bin_exists "$NVHPC_NVCC"
  "$NVHPC_NVCXX" --version
  "$NVHPC_NVCC" --version
  check_size
}

setup_aomp() {
  echo "Preparing AOMP"
  local AOMP_URL="https://github.com/ROCm-Developer-Tools/aomp/releases/download/rel_11.12-0/aomp_Ubuntu1804_11.12-0_amd64.deb"
  # local AOMP_URL="http://0.0.0.0:8000/aomp_Ubuntu1804_11.12-0_amd64.deb"
  get_and_install_deb "aomp" "aomp" "$AOMP_URL"

  export_var AOMP_CXX "$PWD/aomp/usr/lib/aomp/bin/clang++"
  verify_bin_exists "$AOMP_CXX"
  "$AOMP_CXX" --version
  check_size
}

setup_oclcpu() {
  echo "Preparing Intel CPU OpenCL runtime"
  local tarball="oclcpuexp.tar.gz"
  local url="https://github.com/intel/llvm/releases/download/2020-12/oclcpuexp-2020.11.11.0.04_rel.tar.gz"
  # local url="http://localhost:8000/oclcpuexp-2020.11.11.0.04_rel.tar.gz"
  get_and_untar "$tarball" "$url"
  export_var OCL_LIB "$PWD/x64/libOpenCL.so"
  verify_bin_exists "$OCL_LIB"
  check_size
}

setup_kokkos() {
  echo "Preparing Kokkos"
  local kokkos_ver="3.3.01"
  local tarball="kokkos-$kokkos_ver.tar.gz"

  local url="https://github.com/kokkos/kokkos/archive/$kokkos_ver.tar.gz"
  # local url="http://localhost:8000/$kokkos_ver.tar.gz"

  get_and_untar "$tarball" "$url"
  export_var KOKKOS_SRC "$PWD/kokkos-$kokkos_ver"
  verify_dir_exists "$KOKKOS_SRC"
  check_size
}

setup_raja() {
  echo "Preparing RAJA"
  local raja_ver="0.13.0"
  local tarball="raja-$raja_ver.tar.gz"

  local url="https://github.com/LLNL/RAJA/releases/download/v0.13.0/RAJA-v$raja_ver.tar.gz"
  # local url="http://localhost:8000/RAJA-v$raja_ver.tar.gz"

  get_and_untar "$tarball" "$url"
  export_var RAJA_SRC "$PWD/RAJA-v$raja_ver"
  verify_dir_exists "$RAJA_SRC"
  check_size
}

setup_tbb() {
  echo "Preparing TBB"
  local tbb_ver="2021.2.0"
  local tarball="oneapi-tbb-$tbb_ver-lin.tgz"

  local url="https://github.com/oneapi-src/oneTBB/releases/download/v$tbb_ver/oneapi-tbb-$tbb_ver-lin.tgz"
  # local url="http://localhost:8000/oneapi-tbb-$tbb_ver-lin.tgz"

  get_and_untar "$tarball" "$url"
  export_var TBB_LIB "$PWD/oneapi-tbb-$tbb_ver"
  verify_dir_exists "$TBB_LIB"
  check_size
}

setup_clang_gcc() {

  echo "deb http://archive.ubuntu.com/ubuntu focal main universe" | sudo tee -a /etc/apt/sources.list

  sudo apt-get update -qq
  sudo apt-get install -y -qq gcc-10-offload-nvptx gcc-10-offload-amdgcn libtbb2 libtbb-dev g++-10

  export_var GCC_CXX "$(which g++-10)"
  verify_bin_exists "$GCC_CXX"
  "$GCC_CXX" --version

  export_var GCC_STD_PAR_LIB "tbb"
  export_var GCC_OMP_OFFLOAD_AMD true
  export_var GCC_OMP_OFFLOAD_NVIDIA true

  clang++ --version
  export_var CLANG_CXX "$(which clang++)"
  verify_bin_exists "$CLANG_CXX"
  "$CLANG_CXX" --version

  export_var CLANG_STD_PAR_LIB "tbb"
  export_var CLANG_OMP_OFFLOAD_AMD false
  export_var CLANG_OMP_OFFLOAD_NVIDIA false
  check_size

}

setup_rocm() {
  wget -q -O - "https://repo.radeon.com/rocm/rocm.gpg.key" | sudo apt-key add -
  echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/debian/ xenial main' | sudo tee /etc/apt/sources.list.d/rocm.list
  sudo apt-get update -qq
  sudo apt-get install -y -qq rocm-dev
  export_var ROCM_PATH "/opt/rocm"
  export_var HIP_CXX "$ROCM_PATH/bin/hipcc"
  verify_bin_exists "$HIP_CXX"
  "$HIP_CXX" --version
  check_size
}

setup_dpcpp() {

  local nightly="20210106"
  local tarball="dpcpp-$nightly.tar.gz"

  local url="https://github.com/intel/llvm/releases/download/sycl-nightly/$nightly/dpcpp-compiler.tar.gz"
  # local url="http://localhost:8000/dpcpp-compiler.tar.gz"

  get_and_untar "$tarball" "$url"

  export_var DPCPP_DIR "$PWD/dpcpp_compiler/"
  verify_dir_exists "$DPCPP_DIR"
  "$DPCPP_DIR/bin/clang++" --version
  check_size
}

setup_hipsycl() {

  sudo apt-get install -y -qq libboost-fiber-dev libboost-context-dev
  local hipsycl_ver="0.9.0"
  local tarball="v$hipsycl_ver.tar.gz"
  local install_dir="$PWD/hipsycl_dist_$hipsycl_ver"

  local url="https://github.com/illuhad/hipSYCL/archive/v$hipsycl_ver.tar.gz"
  # local url="http://localhost:8000/hipSYCL-$hipsycl_ver.tar.gz"

  get_and_untar "$tarball" "$url"

  if [ "$SETUP" = true ]; then
    local src="$PWD/hipSYCL-$hipsycl_ver"
    rm -rf "$src/build"
    rm -rf "$install_dir"
    cmake "-B$src/build" "-H$src" \
      -DCMAKE_C_COMPILER="$(which gcc-10)" \
      -DCMAKE_CXX_COMPILER="$(which g++-10)" \
      -DCMAKE_INSTALL_PREFIX="$install_dir" \
      -DWITH_ROCM_BACKEND=OFF \
      -DWITH_CUDA_BACKEND=OFF \
      -DWITH_CPU_BACKEND=ON
    cmake --build "$src/build" --target install -j "$(nproc)"
  fi

  export_var HIPSYCL_DIR "$install_dir"
  verify_dir_exists "$HIPSYCL_DIR"
  # note: this will forward --version to the default compiler so it won't say anything about hipsycl
  "$HIPSYCL_DIR/bin/syclcc-clang" --version
  check_size
}

setup_computecpp() {
  echo "TODO ComputeCpp requires registration+login to download"
}

if [ "${GITHUB_ACTIONS:-false}" = true ]; then
  echo "Running in GitHub Actions, defaulting to special export"
  TERM=xterm
  export TERM=xterm
  if [ "$SETUP" = true ]; then
    echo "Deleting extra packages for space in 5 seconds..."
    sleep 5
    echo "Starting apt-get remove:"
    sudo apt-get remove -y azure-cli google-cloud-sdk hhvm google-chrome-stable firefox powershell mono-devel
    sudo apt-get autoremove -y
    check_size
  fi
else
  echo "Running locally, defaulting to standard export"
fi

setup_cmake() {

  echo "Preparing CMake"

  local cmake_release="https://github.com/Kitware/CMake/releases/download"

  get "cmake-3.13.sh" "$cmake_release/v3.13.4/cmake-3.13.4-Linux-x86_64.sh"
  chmod +x "./cmake-3.13.sh" && sh "./cmake-3.13.sh" --skip-license --include-subdir
  export_var CMAKE_3_13_BIN "$PWD/cmake-3.13.4-Linux-x86_64/bin/cmake"
  verify_bin_exists "$CMAKE_3_13_BIN"
  "$CMAKE_3_13_BIN" --version

  get "cmake-3.15.sh" "$cmake_release/v3.15.7/cmake-3.15.7-Linux-x86_64.sh"
  chmod +x "./cmake-3.15.sh" && "./cmake-3.15.sh" --skip-license --include-subdir
  export_var CMAKE_3_15_BIN "$PWD/cmake-3.15.7-Linux-x86_64/bin/cmake"
  verify_bin_exists "$CMAKE_3_15_BIN"
  "$CMAKE_3_15_BIN" --version

  get "cmake-3.18.sh" "$cmake_release/v3.18.6/cmake-3.18.6-Linux-x86_64.sh"
  chmod +x "./cmake-3.18.sh" && "./cmake-3.18.sh" --skip-license --include-subdir
  export_var CMAKE_3_18_BIN "$PWD/cmake-3.18.6-Linux-x86_64/bin/cmake"
  verify_bin_exists "$CMAKE_3_18_BIN"
  "$CMAKE_3_18_BIN" --version

  check_size

}

if [ "$PARALLEL" = true ]; then
  (setup_clang_gcc && setup_rocm && setup_hipsycl) &   # these need apt so run sequentially
  setup_cmake &
  setup_oclcpu &
  setup_aocc &
  setup_nvhpc &
  setup_aomp &
  setup_dpcpp &
  setup_kokkos &
  setup_raja &
  setup_tbb &
  wait
else
  setup_cmake
  setup_aocc
  setup_oclcpu
  setup_nvhpc
  setup_aomp
  setup_dpcpp
  setup_kokkos
  setup_raja
  setup_tbb
  # these need apt
  setup_clang_gcc
  setup_rocm
  setup_hipsycl
fi

echo "Done!"
cd "$PREV_DIR" || exit 1
