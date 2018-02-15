
default: kokkos-stream

include $(KOKKOS_PATH)/Makefile.kokkos

ifndef COMPILER
define compiler_help
Set COMPILER to change flags (defaulting to GNU).
Available compilers are:
  GNU INTEL

endef
$(info $(compiler_help))
COMPILER=GNU
endif

COMPILER_GNU = g++
COMPILER_INTEL = icpc -qopt-streaming-stores=always
CXX = $(COMPILER_$(COMPILER))

ifndef TARGET
define target_help
Set TARGET to change to offload device. Defaulting to CPU.
Available targets are:
  CPU (default)
  GPU
endef
$(info $(target_help))
TARGET=CPU
endif

ifeq ($(TARGET), GPU)
CXX = $(NVCC_WRAPPER)
endif

OBJ = main.o KokkosStream.o

kokkos-stream: $(OBJ) $(KOKKOS_CPP_DEPENDS)
	$(CXX) $(KOKKOS_LDFLAGS) -DKOKKOS -O3 $(EXTRA_FLAGS) $(OBJ) $(KOKKOS_LIBS) -o $@

%.o: %.cpp
	$(CXX) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) -DKOKKOS -O3 $(EXTRA_FLAGS) -c $<

.PHONY: clean
clean:
	rm -f kokkos-stream main.o KokkosStream.o

