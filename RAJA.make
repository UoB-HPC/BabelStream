
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
COMP=$(CXX)
CXXFLAGS = -O3 -std=c++11 -DRAJA_TARGET_CPU

ifndef COMPILER
define compiler_help
Set COMPILER to ensure correct OpenMP flags are set.
Available compilers are:
  INTEL GNU CRAY
endef
$(info $(compiler_help))
endif
ifeq ($(COMPILER), INTEL)
COMP = icpc
CXXFLAGS += -qopenmp
else ifeq ($(COMPILER), GNU)
COMP = g++
CXXFLAGS += -fopenmp
else ifeq ($(COMPILER), CRAY)
COMP = CC
CXXFLAGS +=
endif

else ifeq ($(TARGET), GPU)
COMP = nvcc

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
	$(COMP) $(CXXFLAGS) -DUSE_RAJA -I$(RAJA_PATH)/include $^ $(EXTRA_FLAGS) -L$(RAJA_PATH)/lib -lRAJA -o $@

.PHONY: clean
clean:
	rm -f raja-stream

