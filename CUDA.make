
cuda-stream: main.cpp CUDAStream.cu
	nvcc -std=c++11 -O3 -DCUDA $^ $(EXTRA_FLAGS) -o $@

.PHONY: clean
clean:
	rm -f cuda-stream

