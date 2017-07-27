
COMPUTECPP_FLAGS = $(shell $(COMPUTECPP_PACKAGE_ROOT_DIR)/bin/computecpp_info --dump-device-compiler-flags)

sycl-stream: main.cpp SYCLStream.cpp SYCLStream.sycl
	$(CXX) -O3 -std=c++11 -DSYCL main.cpp SYCLStream.cpp -I$(COMPUTECPP_PACKAGE_ROOT_DIR)/include -include SYCLStream.sycl $(EXTRA_FLAGS) -L$(COMPUTECPP_PACKAGE_ROOT_DIR)/lib -lComputeCpp -lOpenCL -Wl,--rpath=$(COMPUTECPP_PACKAGE_ROOT_DIR)/lib/ -o $@

SYCLStream.sycl: SYCLStream.cpp
	$(COMPUTECPP_PACKAGE_ROOT_DIR)/bin/compute++ SYCLStream.cpp $(COMPUTECPP_FLAGS) -c -I$(COMPUTECPP_PACKAGE_ROOT_DIR)/include -o $@

.PHONY: clean
clean:
	rm -f sycl-stream SYCLStream.sycl SYCLStream.bc
