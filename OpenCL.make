
ifndef COMPILER
define compiler_help
Set COMPILER to change flags (defaulting to GNU).
Available compilers are:
  GNU CLANG INTEL CRAY HCC

endef
$(info $(compiler_help))
COMPILER=GNU
endif

COMPILER_HCC = g++
COMPILER_GNU = g++
COMPILER_CLANG = clang++
COMPILER_INTEL = icpc
COMPILER_CRAY = CC
CXX = $(COMPILER_$(COMPILER))

FLAGS_GNU = -O3 -std=c++11
FLAGS_CLANG = -O3 -std=c++11
FLAGS_HCC = -O3 -std=c++11 -I/opt/rocm/opencl/include/
FLAGS_INTEL = -O3 -std=c++11
FLAGS_CRAY = -O3 -hstd=c++11
CXXFLAGS=$(FLAGS_$(COMPILER))

PLATFORM = $(shell uname -s)
ifeq ($(PLATFORM), Darwin)
  LIBS = -framework OpenCL
else
  LIBS = -lOpenCL
endif

ocl-stream: main.cpp OCLStream.cpp
	$(CXX) $(CXXFLAGS) -DOCL $^ $(EXTRA_FLAGS) $(LIBS) -o $@

.PHONY: clean
clean:
	rm -f ocl-stream

