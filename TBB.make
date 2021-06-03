
ifndef COMPILER
define compiler_help
Set COMPILER to change flags (defaulting to GNU).
Available compilers are:
  GNU INTEL INTEL_LEGACY

endef
$(info $(compiler_help))
COMPILER=GNU
endif


CXX_GNU          = g++
CXX_INTEL        = icpx
CXX_INTEL_LEGACY = icpc
CXX = $(COMPILER_$(COMPILER))

CXXFLAGS_GNU          = -march=native
CXXFLAGS_INTEL        = -march=native
CXXFLAGS_INTEL_LEGACY = -qopt-streaming-stores=always

CXX = $(CXX_$(COMPILER))
CXXFLAGS = -std=c++11 -O3 $(CXXFLAGS_$(COMPILER))



ifndef PARTITIONER
define partitioner_help
Set PARTITIONER to select TBB's partitioner.
Partitioner specifies how a loop template should partition its work among threads.

Available options:
  AUTO     - Optimize range subdivision based on work-stealing events.
  AFFINITY - Proportional splitting that optimizes for cache affinity.
  STATIC   - Distribute work uniformly with no additional load balancing.
  SIMPLE   - Recursively split its range until it cannot be further subdivided.

See https://spec.oneapi.com/versions/latest/elements/oneTBB/source/algorithms.html#partitioners
for more details.

endef
$(info $(partitioner_help))
PARTITIONER=AUTO
endif

PARTITIONER_MODE = -DPARTITIONER_$(PARTITIONER)


tbb-stream: main.cpp TBBStream.cpp
	$(CXX) -DTBB $(PARTITIONER_MODE) $(CXXFLAGS) $^ $(EXTRA_FLAGS) -I$(TBB_DIR)/include -Wl,-rpath,$(TBB_DIR)/lib/intel64/gcc4.8 $(TBB_DIR)/lib/intel64/gcc4.8/libtbb.so -o $@

.PHONY: clean
clean:
	rm -f tbb-stream

