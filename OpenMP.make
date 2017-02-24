
ifndef COMPILER
$(info Define a compiler to set common defaults, i.e make COMPILER=GNU)
endif

ifndef TARGET
$(info No target defined. Specify CPU or GPU. Defaulting to CPU)
TARGET=CPU
endif

COMPILER_ = $(CXX)
COMPILER_GNU = g++
COMPILER_INTEL = icpc
COMPILER_CRAY = CC
COMPILER_CLANG = clang++
CXX = $(COMPILER_$(COMPILER))

FLAGS_ = -O3 -std=c++11
FLAGS_GNU = -O3 -std=c++11
FLAGS_INTEL = -O3 -std=c++11 -xHOST
FLAGS_CRAY = -O3 -hstd=c++11
FLAGS_CLANG = -O3 -std=c++11
CXXFLAGS = $(FLAGS_$(COMPILER))

OMP_ =
OMP_GNU   = -fopenmp
OMP_INTEL = -qopenmp
OMP_CRAY  =
OMP_CLANG = -fopenmp=libomp
OMP = $(OMP_$(COMPILER))

OMP_TARGET_ =
OMP_TARGET_GNU   = -fopenmp
OMP_TARGET_INTEL =
OMP_TARGET_CRAY  =
OMP_TARGET_CLANG = -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda
OMP_TARGET = $(OMP_TARGET_$(COMPILER))

ifeq ($(TARGET), CPU)
OMP = $(OMP_$(COMPILER))
else ifeq ($(TARGET), GPU)
OMP = $(OMP_TARGET_$(COMPILER))
OMP += -DOMP_TARGET_GPU
endif

omp-stream: main.cpp OMPStream.cpp
	$(CXX) $(CXXFLAGS) -DOMP $^ $(OMP) $(EXTRA_FLAGS) -o $@

.PHONY: clean
clean:
	rm -f omp-stream

