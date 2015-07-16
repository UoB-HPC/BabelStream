
LIBS = -l OpenCL

PLATFORM = $(shell uname -s)
ifeq ($(PLATFORM), Darwin)
	LIBS = -framework OpenCL
endif

gpu-stream-ocl: ocl-stream.cpp
	c++ $< -std=c++11 -o $@ $(LIBS)

gpu-stream-cuda: cuda-stream.cu
	nvcc $< -o $@

clean:
	rm -f gpu-stream-ocl
