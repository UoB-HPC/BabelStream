
default: kokkos-stream

include $(KOKKOS_PATH)/Makefile.kokkos

ifndef TARGET
$(info No target defined. Specify CPU or GPU. Defaulting to CPU)
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

