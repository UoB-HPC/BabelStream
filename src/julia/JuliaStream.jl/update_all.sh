#!/bin/bash
# shellcheck disable=SC2034 disable=SC2153

for BACKEND in  "." "AMDGPU" "CUDA" "oneAPI" "Threaded" "KernelAbstractions" 
do
  echo "Updating subproject $BACKEND"
  julia --project="$BACKEND" -e 'import Pkg;  Pkg.resolve(); Pkg.instantiate(); Pkg.update(); Pkg.gc();'
done
