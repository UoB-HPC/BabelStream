
COMPUTECPP_FLAGS = $(shell $(COMPUTECPP_PACKAGE_ROOT_DIR)/bin/computecpp_info --dump-device-compiler-flags)

sycl-stream: main.o SYCLStream.o SYCLStream.sycl
	$(CXX) -O3 -std=c++11 -DSYCL main.o SYCLStream.o $(EXTRA_FLAGS) -L$(COMPUTECPP_PACKAGE_ROOT_DIR)/lib -lComputeCpp -lOpenCL -Wl,--rpath=$(COMPUTECPP_PACKAGE_ROOT_DIR)/lib/ -o $@ 

main.o: main.cpp
	$(CXX) -O3 -std=c++11 -DSYCL main.cpp -c -I$(COMPUTECPP_PACKAGE_ROOT_DIR)/include  $(EXTRA_FLAGS) -o $@ 

SYCLStream.o: SYCLStream.cpp SYCLStream.sycl
	$(CXX) -O3 -std=c++11 -DSYCL SYCLStream.cpp -c -I$(COMPUTECPP_PACKAGE_ROOT_DIR)/include -include SYCLStream.sycl $(EXTRA_FLAGS) -o $@ 

SYCLStream.sycl: SYCLStream.cpp
	$(COMPUTECPP_PACKAGE_ROOT_DIR)/bin/compute++ -DSYCL SYCLStream.cpp $(COMPUTECPP_FLAGS) -c -I$(COMPUTECPP_PACKAGE_ROOT_DIR)/include -o $@ 

.PHONY: clean
clean:
	rm -f sycl-stream SYCLStream.sycl main.o SYCLStream.o
