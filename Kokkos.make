
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

ifeq ($(TARGET), CPU)
TARGET_DEF = -DKOKKOS_TARGET_CPU
else ifeq ($(TARGET), GPU)
CXX = $(NVCC_WRAPPER)
TARGET_DEF =
endif

kokkos-stream: main.cpp KOKKOSStream.cpp $(KOKKOS_CPP_DEPENDS)
	$(CXX) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(KOKKOS_LDFLAGS) main.cpp KOKKOSStream.cpp $(KOKKOS_LIBS) -o $@ -DKOKKOS $(TARGET_DEF) -O3 $(EXTRA_FLAGS)

.PHONY: clean
clean:
	rm -f kokkos-stream

