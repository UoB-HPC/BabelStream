CXXFLAGS?=-O3 -std=c++11

cuda-stream: main.cpp CUDAStream.cu
	nvcc $(CXXFLAGS) -DCUDA $^ $(EXTRA_FLAGS) -o $@

.PHONY: clean
clean:
	rm -f cuda-stream

