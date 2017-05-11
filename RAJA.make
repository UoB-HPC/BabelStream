
ifndef TARGET
define target_help
Set TARGET to change to offload device. Defaulting to CPU.
Available targets are:
  CPU (default)
  GPU
endef
$(info $(target_help))
TARGET=CPU
endif

ifeq ($(TARGET), CPU)

ifndef COMPILER
define compiler_help
Set COMPILER to change flags (defaulting to GNU).
Available compilers are:
  INTEL GNU CRAY XL
endef
$(info $(compiler_help))
COMPILER=GNU
endif

CXX_INTEL = icpc
CXX_GNU   = g++
CXX_CRAY  = CC
CXX_XL    = xlc++

CXXFLAGS_INTEL = -O3 -std=c++11 -qopenmp -xHost -qopt-streaming-stores=always
CXXFLAGS_GNU   = -O3 -std=c++11 -fopenmp
CXXFLAGS_CRAY  = -O3 -hstd=c++11
CXXFLAGS_XL    = -O5 -std=c++11 -qarch=pwr8 -qtune=pwr8 -qsmp=omp -qthreaded

CXX = $(CXX_$(COMPILER))
CXXFLAGS = -DRAJA_TARGET_CPU $(CXXFLAGS_$(COMPILER))

else ifeq ($(TARGET), GPU)
CXX = nvcc

ifndef ARCH
define arch_help
Set ARCH to ensure correct GPU architecture.
Example:
  ARCH=sm_35
endef
$(error $(arch_help))
endif
CXXFLAGS = --expt-extended-lambda -O3 -std=c++11 -x cu -Xcompiler -fopenmp -arch $(ARCH)
endif

raja-stream: main.cpp RAJAStream.cpp
	$(CXX) $(CXXFLAGS) -DUSE_RAJA -I$(RAJA_PATH)/include $^ $(EXTRA_FLAGS) -L$(RAJA_PATH)/lib -lRAJA -o $@

.PHONY: clean
clean:
	rm -f raja-stream

