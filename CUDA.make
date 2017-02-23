
cuda-stream: main.cpp CUDAStream.cu
	nvcc -std=c++11 -O3 -DCUDA $^ -o $@

