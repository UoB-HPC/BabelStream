
default: kokkos-gpu-stream

include $(KOKKOS_PATH)/Makefile.kokkos

kokkos-gpu-stream: main.o KOKKOSStream.o
	$(CXX) $(KOKKOS_LDFLAGS) $^ $(KOKKOS_LIBS) -o $@ -DKOKKOS -O3

%.o:%.cpp $(KOKKOS_CPP_DEPENDS)
	$(NVCC_WRAPPER) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) -c $< -DKOKKOS -O3

.PHONY: clean
clean:
	rm -f main.o KOKKOSStream.o

