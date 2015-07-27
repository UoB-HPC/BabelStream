
LIBS = -l OpenCL
FLAGS = -std=c++11 -O3

PLATFORM = $(shell uname -s)
ifeq ($(PLATFORM), Darwin)
	LIBS = -framework OpenCL
endif

all: gpu-stream-ocl gpu-stream-cuda

gpu-stream-ocl: ocl-stream.cpp
	c++ $< $(FLAGS) -o $@ $(LIBS)

gpu-stream-cuda: cuda-stream.cu
ifeq ($(shell which nvcc > /dev/null; echo $$?), 0)
	nvcc $< $(FLAGS) -o $@
else
	@echo "Cannot find nvcc, please install CUDA";
endif

clean:
	rm -f gpu-stream-ocl gpu-stream-cuda
