
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
	nvcc $< $(FLAGS) -o $@

clean:
	rm -f gpu-stream-ocl gpu-stream-cuda
