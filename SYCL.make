
COMPUTECPP_FLAGS = $(shell $(COMPUTECPP_PREFIX)/bin/computecpp_info --dump-device-compiler-flags)

sycl-stream: main.cpp SYCLStream.cpp SYCLStream.sycl
	$(CXX) -O3 -std=c++11 -DSYCL main.cpp SYCLStream.cpp -I$(COMPUTECPP_PREFIX)/include -include SYCLStream.sycl $(EXTRA_FLAGS) -L$(COMPUTECPP_PREFIX)/lib -lComputeCpp -lOpenCL -Wl,--rpath=$(COMPUTECPP_PREFIX)/lib/ -o $@

SYCLStream.sycl: SYCLStream.cpp
	$(COMPUTECPP_PREFIX)/bin/compute++ SYCLStream.cpp $(COMPUTECPP_FLAGS) -c -I$(COMPUTECPP_PREFIX)/include -o $@

.PHONY: clean
clean:
	rm -f sycl-stream SYCLStream.sycl SYCLStream.bc
