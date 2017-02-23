
COMPUTECPP_FLAGS = $(shell computecpp_info --dump-device-compiler-flags)

sycl-stream: main.cpp SYCLStream.cpp SYCLStream.sycl
	$(CXX) -O3 -std=c++11 -DSYCL main.cpp SYCLStream.cpp -include SYCLStream.sycl $(EXTRA_FLAGS) -lComputeCpp -lOpenCL -o $@

SYCLStream.sycl: SYCLStream.cpp
	compute++ SYCLStream.cpp $(COMPUTECPP_FLAGS) -c

.PHONY: clean
clean:
	rm -f sycl-stream SYCLStream.sycl SYCLStream.bc
