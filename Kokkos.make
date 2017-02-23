
default: kokkos-stream

include $(KOKKOS_PATH)/Makefile.kokkos

ifndef TARGET
$(info No target defined. Specify CPU or GPU. Defaulting to CPU)
TARGET=CPU
endif

ifeq ($(TARGET), CPU)
COMPILER = $(CXX)
TARGET_DEF = -DKOKKOS_TARGET_CPU
else ifeq ($(TARGET), GPU)
COMPILER = $(NVCC_WRAPPER)
TARGET_DEF =
endif

kokkos-stream: main.o KOKKOSStream.o
	$(CXX) $(KOKKOS_LDFLAGS) $^ $(KOKKOS_LIBS) -o $@ -DKOKKOS $(TARGET_DEF) -O3 $(EXTRA_FLAGS)

%.o:%.cpp $(KOKKOS_CPP_DEPENDS)
	$(COMPILER) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) -c $< -DKOKKOS $(TARGET_DEF) -O3 $(EXTRA_FLAGS)

.PHONY: clean
clean:
	rm -f main.o KOKKOSStream.o

