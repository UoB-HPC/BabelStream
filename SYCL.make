ifndef COMPILER
define compiler_help
Set COMPILER to change flags (defaulting to GNU).
Available compilers are:
  HIPSYCL, DPCPP, COMPUTECPP
	

  For HIPSYCL and COMPUTECPP, SYCL_SDK_DIR must be specified, the directory should contain [/lib, /bin, ...]
  For DPCPP, the compiler must be on path
endef
$(info $(compiler_help))
COMPILER=HIPSYCL
endif

ifndef TARGET
define target_help
Set TARGET to change device (defaulting to CPU).
Available targets are:
  CPU AMD NVIDIA

endef
$(info $(target_help))
TARGET=CPU
endif


ifndef ARCH
define arch_help
Set ARCH to change device (defaulting to "").
(GPU *only*) Available targets for HIPSYCL are:
    For CUDA, the architecture has the form sm_XX, e.g. sm_60 for Pascal.
    For ROCm, the architecture has the form gfxYYY, e.g. gfx900 for Vega 10, gfx906 for Vega 20.

endef

ifneq ($(COMPILER), DPCPP)
$(info $(arch_help))
ARCH=

endif

endif

SYCL_COMPUTECPP_SYCLFLAGS = $(shell $(SYCL_SDK_DIR)/bin/computecpp_info --dump-device-compiler-flags)
SYCL_COMPUTECPP_SYCLFLAGS_AMD = $(COMPUTECPP_FLAGS)
SYCL_COMPUTECPP_SYCLFLAGS_CPU = $(COMPUTECPP_FLAGS)
SYCL_COMPUTECPP_SYCLFLAGS_NVIDIA = $(COMPUTECPP_FLAGS) -sycl-target ptx64
SYCL_COMPUTECPP_SYCLCXX = $(SYCL_SDK_DIR)/bin/compute++
SYCL_COMPUTECPP_FLAGS = -O3 --std=c++17
SYCL_COMPUTECPP_LINK_FLAGS = -L$(SYCL_SDK_DIR)/lib -lComputeCpp -lOpenCL -Wl,--rpath=$(SYCL_SDK_DIR)/lib/
SYCL_COMPUTECPP_INCLUDE = -I$(SYCL_SDK_DIR)/include 
SYCL_COMPUTECPP_CXX = g++


SYCL_HIPSYCL_SYCLFLAGS_CPU =    -O3 --std=c++17 --hipsycl-platform=cpu 
SYCL_HIPSYCL_SYCLFLAGS_AMD =    -O3 --std=c++17 --hipsycl-platform=rocm --hipsycl-gpu-arch=$(ARCH)
SYCL_HIPSYCL_SYCLFLAGS_NVIDIA = -O3 --std=c++17 --hipsycl-platform=cuda --hipsycl-gpu-arch=$(ARCH)
SYCL_HIPSYCL_SYCLCXX = $(SYCL_SDK_DIR)/bin/syclcc
SYCL_HIPSYCL_FLAGS = $(SYCL_HIPSYCL_SYCLFLAGS_$(TARGET))
SYCL_HIPSYCL_LINK_FLAGS = -L$(SYCL_SDK_DIR)/lib -Wl,-rpath,$(SYCL_SDK_DIR)/lib
SYCL_HIPSYCL_INCLUDE = 
SYCL_HIPSYCL_CXX = $(SYCL_HIPSYCL_SYCLCXX)


SYCL_DPCPP_SYCLFLAGS_CPU = -O3 --std=c++17
SYCL_DPCPP_SYCLFLAGS_NVIDIA = -O3 --std=c++17 -fsycl -fsycl-targets=nvptx64-nvidia-cuda-sycldevice -fsycl-unnamed-lambda
SYCL_DPCPP_SYCLCXX = dpcpp
SYCL_DPCPP_FLAGS = $(SYCL_DPCPP_SYCLFLAGS_CPU)
SYCL_DPCPP_LINK_FLAGS =  
SYCL_DPCPP_INCLUDE = 
SYCL_DPCPP_CXX = dpcpp


SYCL_SYCLFLAGS = $(SYCL_$(COMPILER)_SYCLFLAGS_$(TARGET))
SYCL_SYCLCXX = $(SYCL_$(COMPILER)_SYCLCXX)

SYCL_FLAGS = $(SYCL_$(COMPILER)_FLAGS)
SYCL_LINK_FLAGS = $(SYCL_$(COMPILER)_LINK_FLAGS)
SYCL_INCLUDE = $(SYCL_$(COMPILER)_INCLUDE)
SYCL_CXX = $(SYCL_$(COMPILER)_CXX)

sycl-stream: main.o SYCLStream.o SYCLStream.sycl
	$(SYCL_CXX) $(SYCL_FLAGS) -DSYCL main.o SYCLStream.o $(EXTRA_FLAGS) $(SYCL_LINK_FLAGS) -o $@ 

main.o: main.cpp
	$(SYCL_CXX) $(SYCL_FLAGS) -DSYCL main.cpp -c $(SYCL_INCLUDE) $(EXTRA_FLAGS) -o $@ 

SYCLStream.o: SYCLStream.cpp SYCLStream.sycl
	$(SYCL_CXX) $(SYCL_FLAGS) -DSYCL SYCLStream.cpp -c $(SYCL_INCLUDE) $(EXTRA_FLAGS) -o $@ 

SYCLStream.sycl: SYCLStream.cpp
	$(SYCL_SYCLCXX) -DSYCL SYCLStream.cpp $(SYCL_SYCLFLAGS) -c $(SYCL_INCLUDE) -o $@ 

.PHONY: clean
clean:
	rm -f sycl-stream SYCLStream.sycl main.o SYCLStream.o
