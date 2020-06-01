
DPCPP_FLAGS=-fsycl -fsycl-targets=nvptx64-nvidia-cuda-sycldevice -fsycl-unnamed-lambda


sycl-stream: main.o SYCLStream.o SYCLStream.sycl
	$(CXX) -O3 -DSYCL main.o SYCLStream.o $(EXTRA_FLAGS) -L$(DPCPP_ROOT_DIR)/lib  -lOpenCL -Wl,--rpath=$(DPCPP_PACKAGE_ROOT_DIR)/lib/ -o $@ 

main.o: main.cpp
	$(CXX) -O3 -std=c++11 -DSYCL main.cpp -c -I$(DPCPP_ROOT_DIR)/include/sycl  $(EXTRA_FLAGS) -o $@ 

SYCLStream.o: SYCLStream.cpp SYCLStream.sycl
	$(CXX) $(DPCPP_FLAGS) -O3 -std=c++11 -DSYCL SYCLStream.cpp -c -I$(DPCPP_ROOT_DIR)/include/sycl  $(EXTRA_FLAGS) -o $@ 

.PHONY: clean
clean:
	rm -f sycl-stream SYCLStream.sycl main.o SYCLStream.o
