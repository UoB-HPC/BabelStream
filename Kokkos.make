
default: kokkos-stream

ifndef DEVICE
define device_help
Set DEVICE to change flags (defaulting to OpenMP).
Available devices are:
  OpenMP, Serial, Pthreads, Cuda, HIP

endef
$(info $(device_help))
DEVICE="OpenMP"
endif
KOKKOS_DEVICES="$(DEVICE)"

ifndef ARCH
define arch_help
Set ARCH to change flags (defaulting to empty).
Available architectures are:
  AMDAVX
  ARMv80 ARMv81 ARMv8-ThunderX
  BGQ Power7 Power8 Power9
  WSM SNB HSW BDW SKX KNC KNL 
  Kepler30 Kepler32 Kepler35 Kepler37 
  Maxwell50 Maxwell52 Maxwell53 
  Pascal60 Pascal61 
  Volta70 Volta72

endef
$(info $(arch_help))
ARCH=""
endif
KOKKOS_ARCH="$(ARCH)"

ifndef COMPILER
define compiler_help
Set COMPILER to change flags (defaulting to GNU).
Available compilers are:
  GNU INTEL CRAY PGI ARMCLANG HIPCC

  Note: you may have to do `export CXX=\path\to\hipcc` in case Kokkos detects the wrong compiler

endef
$(info $(compiler_help))
COMPILER=GNU
endif

COMPILER_ARMCLANG = armclang++
COMPILER_HIPCC = hipcc
COMPILER_GNU = g++
COMPILER_INTEL = icpc -qopt-streaming-stores=always
COMPILER_CRAY = CC
COMPILER_PGI = pgc++
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
ifneq ($(COMPILER), HIPCC)
CXX = $(NVCC_WRAPPER)
endif
endif

OBJ = main.o KokkosStream.o
CXXFLAGS = -O3 
LINKFLAGS = # empty for now



ifeq ($(COMPILER), GNU)
ifeq ($(DEVICE), OpenMP)
CXXFLAGS += -fopenmp
LINKFLAGS += -fopenmp
endif 
endif

include $(KOKKOS_PATH)/Makefile.kokkos

kokkos-stream: $(OBJ) $(KOKKOS_LINK_DEPENDS)
	$(CXX) $(KOKKOS_LDFLAGS) $(LINKFLAGS) $(EXTRA_PATH) $(OBJ) $(KOKKOS_LIBS) $(LIB) -DKOKKOS -o $@

%.o: %.cpp $(KOKKOS_CPP_DEPENDS)
	$(CXX) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(CXXFLAGS) $(EXTRA_INC) -DKOKKOS -c $<

.PHONY: clean
clean:
	rm -f kokkos-stream main.o KokkosStream.o Kokkos_*.o

