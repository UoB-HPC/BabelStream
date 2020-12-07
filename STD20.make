
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

FLAGS_GNU = -O3 -std=c++2a -march=native
CXXFLAGS = $(FLAGS_$(COMPILER))


std20-stream: main.cpp STD20Stream.cpp
	$(CXX) -DSTD20 $(CXXFLAGS) $^ $(EXTRA_FLAGS) -o $@

.PHONY: clean
clean:
	rm -f std20-stream

