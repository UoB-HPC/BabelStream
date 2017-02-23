
ifndef COMPILER
$(info Define a compiler to set common defaults, i.e make COMPILER=GNU)
endif

COMPILER_ = $(CXX)
COMPILER_GNU = g++
COMPILER_INTEL = icpc
COMPILER_CRAY = CC
CC = $(COMPILER_$(COMPILER))

FLAGS_ = -O3
FLAGS_GNU = -O3 -std=c++11
FLAGS_INTEL = -O3 -std=c++11
FLAGS_CRAY = -O3 -hstd=c++11
CFLAGS = $(FLAGS_$(COMPILER))

OMP_ = 
OMP_GNU   = -fopenmp
OMP_INTEL = -qopenmp
OMP_CRAY  =
OMP = $(OMP_$(COMPILER))

omp-stream: main.cpp OMPStream.cpp
	$(CC) -O3 -std=c++11 -DOMP $^ $(OMP) -o $@

omp-target-stream: main.cpp OMPStream.cpp
	$(CC) -O3 -std=c++11 -DOMP -DOMP_TARGET_GPU $^ $(OMP) -o $@

