#!/usr/bin/env bash
set -euo pipefail

apt_options=(-o Acquire::Retries=5 -o Acquire::http::Timeout=30 -o Acquire::https::Timeout=30)
curl_options=(--fail --silent --show-error --location --retry 5 --retry-all-errors --retry-delay 30 --connect-timeout 30)

apt-get "${apt_options[@]}" update
apt-get "${apt_options[@]}" install -y --no-install-recommends cmake ninja-build g++ curl ca-certificates gpg

if [[ -n ${CI_APT_SOURCE:-} ]]; then
  printf 'Downloading repository key: %s\n' "$CI_APT_KEY"
  curl "${curl_options[@]}" "$CI_APT_KEY" |
    gpg --dearmor --yes -o /usr/share/keyrings/babelstream-toolchain.gpg
  printf 'Adding package repository: %s\n' "$CI_APT_SOURCE"
  printf 'deb [signed-by=/usr/share/keyrings/babelstream-toolchain.gpg] %s\n' \
    "$CI_APT_SOURCE" > /etc/apt/sources.list.d/babelstream-toolchain.list
  # XXX Keep vendor packages together (Ubuntu's hipcc has a higher version number)
  repo_host=${CI_APT_SOURCE#*://}
  repo_host=${repo_host%%/*}
  printf 'Package: *\nPin: origin %s\nPin-Priority: 600\n' "$repo_host" \
    > /etc/apt/preferences.d/babelstream-toolchain
  apt-get "${apt_options[@]}" update
fi

read -r -a packages <<< "$CI_PACKAGES"
if ((${#packages[@]})); then
  apt-get "${apt_options[@]}" install -y --no-install-recommends "${packages[@]}"
fi

if [[ -n ${CI_KOKKOS_REF:-} ]]; then
  printf 'Cloning Kokkos %s from https://github.com/kokkos/kokkos.git\n' "$CI_KOKKOS_REF"
  git clone --depth 1 --branch "$CI_KOKKOS_REF" https://github.com/kokkos/kokkos.git /opt/kokkos
fi

if [[ -n ${CI_RAJA_REF:-} ]]; then
  printf 'Cloning RAJA %s from https://github.com/LLNL/RAJA.git\n' "$CI_RAJA_REF"
  git clone --depth 1 --recurse-submodules --shallow-submodules --branch "$CI_RAJA_REF" \
    https://github.com/LLNL/RAJA.git /opt/raja
fi

if [[ -n ${CI_ADAPTIVECPP_SOURCE_URL:-} ]]; then
  acpp_build=$(mktemp -d)
  printf 'Downloading AdaptiveCpp: %s\n' "$CI_ADAPTIVECPP_SOURCE_URL"
  curl "${curl_options[@]}" "$CI_ADAPTIVECPP_SOURCE_URL" -o "$acpp_build/source.tar.gz"
  echo "$CI_ADAPTIVECPP_SHA256" " $acpp_build/source.tar.gz" | sha256sum --check
  tar -xzf "$acpp_build/source.tar.gz" -C "$acpp_build" --strip-components=1
  cmake -S "$acpp_build" -B "$acpp_build/build" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/adaptivecpp \
    "-DCMAKE_C_COMPILER=$CC" "-DCMAKE_CXX_COMPILER=$CXX" \
    -DLLVM_DIR=/usr/lib/llvm-18/lib/cmake/llvm -DCLANG_EXECUTABLE_PATH=/usr/bin/clang++-18 \
    -DWITH_CUDA_BACKEND=OFF -DWITH_ROCM_BACKEND=OFF \
    -DWITH_OPENCL_BACKEND=OFF -DWITH_LEVEL_ZERO_BACKEND=OFF
  cmake --build "$acpp_build/build" --parallel "${CMAKE_BUILD_PARALLEL_LEVEL:-2}"
  cmake --install "$acpp_build/build"
  rm -rf "$acpp_build"
fi

if [[ -n ${CI_FUTHARK_REF:-} ]]; then
  futhark_build=$(mktemp -d)
  futhark_version=${CI_FUTHARK_REF#v}
  printf 'Downloading Futhark %s\n' "$CI_FUTHARK_REF"
  curl "${curl_options[@]}" \
    "https://github.com/diku-dk/futhark/releases/download/${CI_FUTHARK_REF}/futhark-${futhark_version}-linux-x86_64.tar.xz" \
    -o "$futhark_build/futhark.tar.xz"
  echo "$CI_FUTHARK_SHA256" " $futhark_build/futhark.tar.xz" | sha256sum --check
  mkdir -p /opt/futhark
  tar -xJf "$futhark_build/futhark.tar.xz" -C /opt/futhark --strip-components=1
  rm -rf "$futhark_build"
fi

"$CC" --version
"$CXX" --version
cmake --version
