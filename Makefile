LDLIBS = -l OpenCL
CXXFLAGS = -std=c++11 -O3

PLATFORM = $(shell uname -s)
ifeq ($(PLATFORM), Darwin)
	LDLIBS = -framework OpenCL
endif

all: gpu-stream-ocl gpu-stream-cuda gpu-stream-hip


gpu-stream-ocl: ocl-stream.cpp common.o Makefile
	$(CXX) $(CXXFLAGS) -Wno-deprecated-declarations common.o $< -o $@ $(LDLIBS)

common.o: common.cpp common.h Makefile

gpu-stream-cuda: cuda-stream.cu common.o Makefile
ifeq ($(shell which nvcc > /dev/null; echo $$?), 0)
	nvcc $(CXXFLAGS) common.o $< -o $@
else
	$(error "Cannot find nvcc, please install CUDA toolkit")
endif
HIP_PATH?=../../..
HIPCC=$(HIP_PATH)/bin/hipcc

hip-stream.o : hip-stream.cpp
	$(HIPCC) $(CXXFLAGS) -c $< -o $@

gpu-stream-hip: hip-stream.o common.o Makefile
ifeq ($(shell which $(HIPCC) > /dev/null; echo $$?), 0)
	$(HIPCC) $(CXXFLAGS) common.o $< -lm -o $@
else
	$(error "Cannot find $(HIPCC), please install HIP toolkit")
endif


.PHONY: clean

clean:
	rm -f gpu-stream-ocl gpu-stream-cuda gpu-stream-hip *.o

