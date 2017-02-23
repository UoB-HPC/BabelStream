
sycl-stream: main.cpp SYCLStream.cpp SYCLStream.sycl
	$(CXX) -O3 -std=c++11 -DSYCL main.cpp SYCLStream.cpp -include SYCLStream.sycl $(EXTRA_FLAGS) -lComputeCpp -lOpenCL -o $@


SYCLStream.sycl: SYCLStream.cpp
	compute++ SYCLStream.cpp -sycl -no-serial-memop -O2 -emit-llvm -c

.PHONY: clean
clean:
	rm -f SYCLStream.sycl
