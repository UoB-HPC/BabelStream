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

ifeq ($(COMPILER), HIPSYCL)
ifneq ($(TARGET), CPU)
$(info $(arch_help))
ARCH=
endif
endif

endif

SYCL_COMPUTECPP_SYCLFLAGS = $(shell $(SYCL_SDK_DIR)/bin/computecpp_info --dump-device-compiler-flags) -no-serial-memop -sycl-driver
SYCL_COMPUTECPP_SYCLFLAGS_CPU = $(SYCL_COMPUTECPP_SYCLFLAGS)
SYCL_COMPUTECPP_SYCLFLAGS_AMD = $(SYCL_COMPUTECPP_SYCLFLAGS)
SYCL_COMPUTECPP_SYCLFLAGS_NVIDIA = $(SYCL_COMPUTECPP_SYCLFLAGS) -sycl-target ptx64
SYCL_COMPUTECPP_SYCLCXX = $(SYCL_SDK_DIR)/bin/compute++
SYCL_COMPUTECPP_FLAGS = -O3 -std=c++17
SYCL_COMPUTECPP_LINK_FLAGS = -Wl,-rpath=$(SYCL_SDK_DIR)/lib/ $(SYCL_SDK_DIR)/lib/libComputeCpp.so -lOpenCL 
SYCL_COMPUTECPP_INCLUDE = -I$(SYCL_SDK_DIR)/include 

SYCL_HIPSYCL_SYCLFLAGS_CPU =    --hipsycl-platform=cpu
SYCL_HIPSYCL_SYCLFLAGS_AMD =    --hipsycl-platform=rocm --hipsycl-gpu-arch=$(ARCH)
SYCL_HIPSYCL_SYCLFLAGS_NVIDIA = --hipsycl-platform=cuda --hipsycl-gpu-arch=$(ARCH)
SYCL_HIPSYCL_SYCLCXX = $(SYCL_SDK_DIR)/bin/syclcc
SYCL_HIPSYCL_FLAGS = -O3 --std=c++17
SYCL_HIPSYCL_LINK_FLAGS = -L$(SYCL_SDK_DIR)/lib -Wl,-rpath,$(SYCL_SDK_DIR)/lib
SYCL_HIPSYCL_INCLUDE = 

SYCL_DPCPP_SYCLFLAGS_NVIDIA = -fsycl -fsycl-targets=nvptx64-nvidia-cuda-sycldevice -fsycl-unnamed-lambda
SYCL_DPCPP_SYCLCXX = dpcpp
SYCL_DPCPP_FLAGS = -O3 --std=c++17
SYCL_DPCPP_LINK_FLAGS =  
SYCL_DPCPP_INCLUDE = 


SYCL_SYCLFLAGS = $(SYCL_$(COMPILER)_SYCLFLAGS_$(TARGET))
SYCL_SYCLCXX = $(SYCL_$(COMPILER)_SYCLCXX)
SYCL_FLAGS = $(SYCL_$(COMPILER)_FLAGS)
SYCL_LINK_FLAGS = $(SYCL_$(COMPILER)_LINK_FLAGS)
SYCL_INCLUDE = $(SYCL_$(COMPILER)_INCLUDE)

# only ComputeCpp generates .sycl files which is a bit odd to deal with so we opted to compile everything together
sycl-stream: main.cpp SYCLStream.cpp
	$(SYCL_SYCLCXX) $(SYCL_SYCLFLAGS) $(SYCL_FLAGS) $(SYCL_INCLUDE) -DSYCL $(EXTRA_FLAGS) $(SYCL_LINK_FLAGS) $^ -o $@

.PHONY: clean
clean:
	rm -f sycl-stream
