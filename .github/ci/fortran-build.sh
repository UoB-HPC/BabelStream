#!/usr/bin/env bash
set -euo pipefail

if [[ -n ${CI_ENV_SCRIPT:-} ]]; then
  set +u # Vendor environment scripts may reference unset variables.
  # shellcheck disable=SC1090
  source "$CI_ENV_SCRIPT"
  set -u
fi

read -r -a make_args <<< "${CI_MAKE_ARGS:-}"

for precision in double float; do
  build_dir="$CI_BUILD_DIR/$precision"
  rm -rf "$build_dir"
  mkdir -p "$build_dir"
  cp -r src/fortran/. "$build_dir"

  extra_flags=""
  if [[ $precision == float ]]; then extra_flags="-DUSE_FLOAT"; fi

  echo "::group::$CI_IMPL ($precision)"
  make -C "$build_dir" "COMPILER=$CI_COMPILER" "IMPLEMENTATION=$CI_IMPL" "FC=$FC" \
    "EXTRA_FCFLAGS=$extra_flags" "${make_args[@]}"
  if [[ -n ${CI_SKIP_RUN:-} ]]; then
    readelf -SW "$build_dir/BabelStream.$CI_COMPILER.$CI_IMPL" | grep -Eq '\.(nv_fatbin|gnu\.offload_funcs)' ||
      { echo "no device code in the binary"; exit 1; }
  fi
  if [[ -z ${CI_SKIP_RUN:-} ]]; then
    exe="$build_dir/BabelStream.$CI_COMPILER.$CI_IMPL"
    for mode in "" --triad-only --nstream-only; do
      # shellcheck disable=SC2086
      "$exe" --arraysize 100003 $mode
    done
  fi
  echo "::endgroup::"
done
