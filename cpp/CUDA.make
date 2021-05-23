CXXFLAGS=-O3
CUDA_CXX=nvcc


ifndef NVARCH
define nvarch_help
Set NVARCH to select sm_?? version.
Default: sm_60

endef
$(info $(nvarch_help))
NVARCH=sm_60
endif


ifndef MEM
define mem_help
Set MEM to select memory mode.
Available options:
  DEFAULT   - allocate host and device memory pointers.
  MANAGED   - use CUDA Managed Memory.
  PAGEFAULT - shared memory, only host pointers allocated.

endef
$(info $(mem_help))
MEM=DEFAULT
endif

MEM_MANAGED= -DMANAGED
MEM_PAGEFAULT= -DPAGEFAULT
MEM_MODE = $(MEM_$(MEM))


cuda-stream: main.cpp CUDAStream.cu
	$(CUDA_CXX) -std=c++11 $(CXXFLAGS) -arch=$(NVARCH) $(MEM_MODE) -DCUDA $^ $(EXTRA_FLAGS) -o $@

.PHONY: clean
clean:
	rm -f cuda-stream

