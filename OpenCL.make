
ifndef COMPILER
$(info Define a compiler to set common defaults, i.e make COMPILER=GNU)
endif

COMPILER_ = $(CXX)
COMPILER_GNU = g++
COMPILER_CRAY = CC

FLAGS_ = -O3 -std=c++11
FLAGS_GNU = -O3 -std=c++11
FLAGS_CRAY = -O3 -hstd=c++11
CXXFLAGS=$(FLAGS_$(COMPILER))

ocl-stream: main.cpp OCLStream.cpp
	$(COMPILER_$(COMPILER)) $(CXXFLAGS) -DOCL $^ $(EXTRA_FLAGS) -lOpenCL -o $@

.PHONY: clean
clean:
	rm -f ocl-stream

