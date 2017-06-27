CXXFLAGS=-O3
CUDA_CXX=nvcc
CUDA_LIBS=-lcudart

cuda-stream: main.cpp CUDAStream.o
	$(CXX) -std=c++11 $(CXXFLAGS) -DCUDA $^ $(EXTRA_FLAGS) $(CUDA_LIBS) -o $@

CUDAStream.o: CUDAStream.cu
	$(CUDA_CXX) -std=c++11 $(CXXFLAGS) $< $(CUDA_EXTRA_FLAGS) -c

.PHONY: clean
clean:
	rm -f cuda-stream CUDAStream.o

