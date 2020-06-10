ifndef COMPILER
define compiler_help
Set COMPILER to change flags (defaulting to GNU).
Available compilers are:
  GNU

endef
$(info $(compiler_help))
COMPILER=GNU
endif

COMPILER_GNU = g++
CXX = $(COMPILER_$(COMPILER))

FLAGS_ = -O3 -std=c++17
FLAGS_GNU = -O3 -std=c++17
CXXFLAGS=$(FLAGS_$(COMPILER))

PLATFORM = $(shell uname -s)
LIBS = -lpoplar -lpopops

poplar-stream: main.cpp PoplarStream.cpp 
	$(CXX) $(CXXFLAGS) -DPOPLAR $^ $(EXTRA_FLAGS) $(LIBS) -o $@

.PHONY: clean
clean:
	rm -f poplar-stream PoplarKernels.gc
