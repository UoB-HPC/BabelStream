
ifndef TARGET
$(info No target defined. Specify CPU or GPU. Defaulting to CPU)
TARGET=CPU
endif

ifeq ($(TARGET), CPU)
COMP=$(CXX)
CXXFLAGS = -O3 -std=c++11 -DRAJA_TARGET_CPU

ifndef COMPILER
$(error No COMPILER defined. Specify COMPILER for correct OpenMP flag.)
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
$(error No ARCH defined. Specify target GPU architecture (e.g. ARCH=sm_35))
endif
CXXFLAGS = --expt-extended-lambda -O3 -std=c++11 -x cu -Xcompiler -fopenmp -arch $(ARCH)
endif

raja-stream: main.cpp RAJAStream.cpp
	$(COMP) $(CXXFLAGS) -DUSE_RAJA -I$(RAJA_PATH)/include $^ $(EXTRA_FLAGS) -L$(RAJA_PATH)/lib -lRAJA -o $@
