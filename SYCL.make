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

SYCL_COMPUTECPP_FLAGS = $(shell $(SYCL_SDK_DIR)/bin/computecpp_info --dump-device-compiler-flags)
SYCL_COMPUTECPP_FLAGS_AMD = $(COMPUTECPP_FLAGS)
SYCL_COMPUTECPP_FLAGS_CPU = $(COMPUTECPP_FLAGS)
SYCL_COMPUTECPP_FLAGS_NVIDIA = $(COMPUTECPP_FLAGS) -sycl-target ptx64
SYCL_COMPUTECPP_LINK_FLAGS = -L$(SYCL_SDK_DIR)/lib -lComputeCpp -lOpenCL -Wl,--rpath=$(SYCL_SDK_DIR)/lib/
SYCL_COMPUTECPP_INCLUDE = -I$(SYCL_SDK_DIR)/include 
SYCL_COMPUTECPP_SYCL_CXX = $(SYCL_SDK_DIR)/bin/compute++
SYCL_COMPUTECPP_CXX_CXX = g++

SYCL_HIPSYCL_FLAGS_CPU = -O3 --hipsycl-platform=cpu 
SYCL_HIPSYCL_FLAGS_AMD = -O3 --hipsycl-platform=rocm --hipsycl-gpu-arch=$(ARCH)
SYCL_HIPSYCL_FLAGS_NVIDIA = -O3 --hipsycl-platform=cuda --hipsycl-gpu-arch=$(ARCH)
SYCL_HIPSYCL_LINK_FLAGS = -L$(SYCL_SDK_DIR)/lib -Wl,-rpath,$(SYCL_SDK_DIR)/lib
SYCL_HIPSYCL_INCLUDE = 
SYCL_HIPSYCL_SYCL_CXX = $(SYCL_SDK_DIR)/bin/syclcc
SYCL_HIPSYCL_CXX_CXX = $(SYCL_HIPSYCL_SYCL_CXX)

SYCL_DPCPP_FLAGS_CPU = -O3 
SYCL_DPCPP_LINK_FLAGS =  
SYCL_DPCPP_INCLUDE = 
SYCL_DPCPP_SYCL_CXX = dpcpp
SYCL_DPCPP_CXX_CXX = dpcpp


SYCL_FLAGS = $(SYCL_$(COMPILER)_FLAGS_$(TARGET))
SYCL_LINK_FLAGS = $(SYCL_$(COMPILER)_LINK_FLAGS)
SYCL_INCLUDE = $(SYCL_$(COMPILER)_INCLUDE)
SYCL_SYCL_CXX = $(SYCL_$(COMPILER)_SYCL_CXX)
SYCL_CXX_CXX = $(SYCL_$(COMPILER)_CXX_CXX)

sycl-stream: main.o SYCLStream.o SYCLStream.sycl
	$(SYCL_CXX_CXX) -O3 -std=c++17 -DSYCL main.o SYCLStream.o $(EXTRA_FLAGS) $(SYCL_LINK_FLAGS) -o $@ 

main.o: main.cpp
	$(SYCL_CXX_CXX) -O3 -std=c++17 -DSYCL main.cpp -c $(SYCL_INCLUDE) $(EXTRA_FLAGS) -o $@ 

SYCLStream.o: SYCLStream.cpp SYCLStream.sycl
	$(SYCL_CXX_CXX) -O3 -std=c++17 -DSYCL SYCLStream.cpp -c $(SYCL_INCLUDE) $(EXTRA_FLAGS) -o $@ 

SYCLStream.sycl: SYCLStream.cpp
	$(SYCL_SYCL_CXX) -DSYCL SYCLStream.cpp $(SYCL_FLAGS) -c $(SYCL_INCLUDE) -o $@ 

.PHONY: clean
clean:
	rm -f sycl-stream SYCLStream.sycl main.o SYCLStream.o
