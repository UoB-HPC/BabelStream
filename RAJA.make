
ifndef TARGET
$(info No target defined. Specify CPU or GPU. Defaulting to CPU)
TARGET=CPU
endif

ifeq ($(TARGET), CPU)
COMP=$(CXX)
CFLAGS = -O3 -std=c++11 -DRAJA_TARGET_CPU 

ifndef COMPILER
$(error No COMPILER defined. Specify COMPILER for correct OpenMP flag.)
endif
ifeq ($(COMPILER), INTEL)
COMP = icpc
CFLAGS += -qopenmp
else ifeq ($(COMPILER), GNU)
COMP = g++
CFLAGS += -fopenmp
else ifeq ($(COMPILER), CRAY)
COMP = CC
CFLAGS +=
endif

else ifeq ($(TARGET), GPU)
COMP = nvcc
CFLAGS = --expt-extended-lambda -O3 -std=c++11 -x cu -Xcompiler -fopenmp
endif

raja-stream: main.cpp RAJAStream.cpp
	$(COMP) $(CFLAGS) -DUSE_RAJA -I$(RAJA_PATH)/include $^ $(EXTRA_FLAGS) -L$(RAJA_PATH)/lib -lRAJA -o $@

