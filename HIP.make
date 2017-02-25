
# TODO: HIP with HCC

HIPCC = hipcc

ifndef CUDA_PATH
ifeq (,$(wildcard /usr/local/bin/nvcc))
$(error /usr/local/bin/nvcc not found, set CUDA_PATH instead)
endif
endif

hip-stream: main.cpp HIPStream.cu
	$(HIPCC) $(CXXFLAGS) -std=c++11 -DHIP $^ $(EXTRA_FLAGS) -o $@

.PHONY: clean
clean:
	rm -f hip-stream

