
ifndef COMPILER
define compiler_help
Set COMPILER to change flags (defaulting to GNU).
Available compilers are:
  INTEL

endef
$(info $(compiler_help))
COMPILER=INTEL
endif

COMPILER_INTEL=icpc
CXX=$(COMPILER_$(COMPILER))

FLAGS_INTEL=-O3 -std=c++17 -qopenmp-simd -tbb -xHost -qstreaming-stores=always
CXXFLAGS=$(FLAGS_$(COMPILER))

stl-stream: main.cpp STLStream.cpp
	$(CXX) $(CXXFLAGS) -DSTL $^ $(EXTRA_FLAGS) -o $@

.PHONY: clean
clean:
	rm -f stl-stream

