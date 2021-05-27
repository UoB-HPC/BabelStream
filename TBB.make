
ifndef COMPILER
define compiler_help
Set COMPILER to change flags (defaulting to GNU).
Available compilers are:
  GNU

endef
$(info $(compiler_help))
COMPILER=GNU
endif

TBB_LIB=

COMPILER_GNU = g++
CXX = $(COMPILER_$(COMPILER))

FLAGS_GNU = -O3 -std=c++14 -march=native
CXXFLAGS = $(FLAGS_$(COMPILER))


tbb-stream: main.cpp TBBStream.cpp
	$(CXX) -DTBB $(CXXFLAGS) $^ $(EXTRA_FLAGS) -I$(TBB_DIR)/include -Wl,-rpath,$(TBB_DIR)/lib/intel64/gcc4.8 $(TBB_DIR)/lib/intel64/gcc4.8/libtbb.so -o $@

.PHONY: clean
clean:
	rm -f tbb-stream

