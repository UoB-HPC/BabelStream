
ifndef COMPILER
define compiler_help
Set COMPILER to change flags (defaulting to GNU).
Available compilers are:
  CLANG CRAY GNU GNU_PPC INTEL XL PGI
  NEC ARMCLANG AOMP FUJITSU

Note: GCC on PPC requires -mcpu=native instead of -march=native so we have a special case for it

endef
$(info $(compiler_help))
COMPILER=GNU
endif

ifndef TARGET
define target_help
Set TARGET to change device (defaulting to CPU).
Available targets are:
  CPU NVIDIA AMD INTEL_GPU

endef
$(info $(target_help))
TARGET=CPU
endif

ifeq ("$(COMPILER)", "CLANG")
  ifdef TARGET
    ifeq ("$(TARGET)", "NVIDIA")
      ifndef NVARCH
        define nvarch_help
        Set NVARCH to select sm_?? version.
        Default: sm_60

        endef
        $(info $(nvarch_help))
        NVARCH=sm_60
      endif
    endif
  endif
endif

COMPILER_ARMCLANG = armclang++
COMPILER_GNU = g++
COMPILER_GNU_PPC = g++
COMPILER_INTEL = icpc
COMPILER_CRAY = CC
COMPILER_CLANG = clang++
COMPILER_XL = xlc++
COMPILER_PGI = pgc++
COMPILER_NEC = /opt/nec/ve/bin/nc++
COMPILER_AOMP = clang++
COMPILER_FUJITSU=FCC
CXX = $(COMPILER_$(COMPILER))

FLAGS_GNU = -O3 -std=c++11 -march=native
FLAGS_GNU_PPC = -O3 -std=c++11 -mcpu=native
FLAGS_INTEL = -O3 -std=c++11
FLAGS_CRAY = -O3 -std=c++11
FLAGS_CLANG = -O3 -std=c++11
FLAGS_XL = -O5 -qarch=auto -qtune=auto -std=c++11
FLAGS_PGI = -O3 -std=c++11
FLAGS_NEC = -O4 -finline -std=c++11
FLAGS_ARMCLANG = -O3 -std=c++11
FLAGS_AOMP = -O3 -std=c++11
FLAGS_FUJITSU=-Kfast -std=c++11 -KA64FX -KSVE -KARMV8_3_A -Kzfill=100 -Kprefetch_sequential=soft -Kprefetch_line=8 -Kprefetch_line_L2=16
CXXFLAGS = $(FLAGS_$(COMPILER))

# OpenMP flags for CPUs
OMP_ARMCLANG_CPU   = -fopenmp
OMP_GNU_CPU   = -fopenmp
OMP_GNU_PPC_CPU = -fopenmp
OMP_INTEL_CPU = -qopenmp
OMP_CRAY_CPU  = -fopenmp
OMP_CLANG_CPU = -fopenmp=libomp
OMP_XL_CPU = -qsmp=omp -qthreaded
OMP_PGI_CPU = -mp
OMP_NEC_CPU = -fopenmp
OMP_FUJITSU_CPU=-Kopenmp

# OpenMP flags for NVIDIA
OMP_CRAY_NVIDIA  = -DOMP_TARGET_GPU
OMP_CLANG_NVIDIA = -DOMP_TARGET_GPU -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=$(NVARCH)
OMP_GNU_NVIDIA = -DOMP_TARGET_GPU -fopenmp -foffload=nvptx-none
OMP_GNU_AMD = -DOMP_TARGET_GPU -fopenmp -foffload=amdgcn-amdhsa

OMP_INTEL_CPU = -xHOST -qopt-streaming-stores=always -qopenmp
OMP_INTEL_INTEL_GPU = -DOMP_TARGET_GPU -qnextgen -fiopenmp -fopenmp-targets=spir64

OMP_AOMP_GPU = -DOMP_TARGET_GPU -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx906

ifndef OMP_$(COMPILER)_$(TARGET)
$(error Targeting $(TARGET) with $(COMPILER) not supported)
endif

OMP = $(OMP_$(COMPILER)_$(TARGET))

omp-stream: main.cpp OMPStream.cpp
	$(CXX) $(CXXFLAGS) -DOMP $^ $(OMP) $(EXTRA_FLAGS) -o $@

.PHONY: clean
clean:
	rm -f omp-stream
