
ifndef COMPILER
$(info Define a compiler to set common defaults, i.e make COMPILER=GNU)
endif

COMPILER_ = $(CXX)
COMPILER_PGI = pgc++
COMPILER_CRAY = CC
CXX = $(COMPILER_$(COMPILER))

FLAGS_ = -O3

FLAGS_PGI = -std=c++11 -O3 -acc
ifeq ($(COMPILER), PGI)
ifndef TARGET
$(info Set a TARGET to ensure PGI targets the correct offload device. i.e. TARGET=GPU or CPU)
endif
endif
ifeq ($(TARGET), GPU)
FLAGS_PGI += -ta=nvidia
else ifeq ($(TARGET), CPU)
FLAGS_PGI += -ta=multicore
endif

FLAGS_CRAY = -hstd=c++11
CFLAGS = $(FLAGS_$(COMPILER))

acc-stream: main.cpp ACCStream.cpp
	$(CXX) $(CFLAGS) -DACC $^ $(EXTRA_FLAGS) -o $@

