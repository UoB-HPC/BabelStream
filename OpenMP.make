
ifndef COMPILER
$(info Define a compiler to set common defaults, i.e make COMPILER=GNU)
endif

COMPILER_ = $(CXX)
COMPILER_GNU = g++
COMPILER_INTEL = icpc
COMPILER_CRAY = CC
COMPILER_CLANG = clang++
CXX = $(COMPILER_$(COMPILER))

FLAGS_ = -O3
FLAGS_GNU = -O3 -std=c++11
FLAGS_INTEL = -O3 -std=c++11
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
OMP_TARGET_CLANG = -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda
OMP_TARGET = $(OMP_TARGET_$(COMPILER))

omp-stream: main.cpp OMPStream.cpp
	$(CXX) $(CXXFLAGS) -DOMP $^ $(OMP) $(EXTRA_FLAGS) -o $@

omp-target-stream: main.cpp OMPStream.cpp
	$(CXX) $(CXXFLAGS) -DOMP -DOMP_TARGET_GPU $^ $(OMP_TARGET) $(EXTRA_FLAGS) -o $@

