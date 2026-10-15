#!/usr/bin/env bash
set -euo pipefail

if [[ -n ${CI_ENV_SCRIPT:-} ]]; then
  set +u # Vendor environment scripts may reference unset variables.
  # shellcheck disable=SC1090
  source "$CI_ENV_SCRIPT"
  set -u
fi

read -r -a model_flags <<< "$CI_CMAKE_ARGS"
native_flags=()
if [[ -n ${ACT:-} ]]; then
  native_flags=("-DCMAKE_CXX_FLAGS=$CI_NATIVE_CXX_FLAGS")
  if [[ -n ${CI_NATIVE_CUDA_FLAGS:-} ]]; then
    native_flags+=("-DCMAKE_CUDA_FLAGS=$CI_NATIVE_CUDA_FLAGS")
  fi
fi

cmake -S . -B "$CI_BUILD_DIR" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release "-DBUILD_TESTING=$BUILD_TESTING" -DUSAGE=OFF \
  "-DCMAKE_C_COMPILER=$CC" "-DCMAKE_CXX_COMPILER=$CXX" \
  "-DMODEL=$CI_MODEL" "${native_flags[@]}" "${model_flags[@]}"
cmake --build "$CI_BUILD_DIR"
if [[ $BUILD_TESTING == OFF ]]; then
  readelf -SW "$CI_BUILD_DIR"/*-stream | grep -Eq '(\.(nv_fatbin|hip_fatbin|gnu\.offload_funcs|tgtimg)|llvm_offload_entries)' ||
    { echo "no device code in the binary"; exit 1; }
fi
if [[ $BUILD_TESTING == ON ]]; then
  ctest --test-dir "$CI_BUILD_DIR" --output-on-failure --no-tests=error
fi
