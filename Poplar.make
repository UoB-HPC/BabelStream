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

FLAGS_ = -O3 -std=c++17 -Wall
FLAGS_GNU = -O3 -std=c++17 -Wall
CXXFLAGS=$(FLAGS_$(COMPILER))

PLATFORM = $(shell uname -s)
LIBS = -lpoplar -lpopops -lpoputil


.PHONY: all
all: poplar-stream popops-stream poplar-stream-vectorised

.PHONY: clean
clean:
	rm -f poplar-stream popops-stream poplar-stream-vectorised PoplarKernels.gc

poplar-stream: main.cpp PoplarStream.cpp 
	$(CXX) $(CXXFLAGS) -DPOPLAR $^ $(EXTRA_FLAGS) $(LIBS) -o $@

poplar-stream-vectorised: main.cpp PoplarStream.cpp 
	$(CXX) $(CXXFLAGS) -DPOPLAR -DVECTORISED=true $^ $(EXTRA_FLAGS) $(LIBS) -o $@	

popops-stream: main.cpp PopopsStream.cpp 
	$(CXX) $(CXXFLAGS) -DPOPLAR $^ $(EXTRA_FLAGS) $(LIBS) -o $@


