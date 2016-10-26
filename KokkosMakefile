
default: gpu-stream-kokkos

include $(KOKKOS_PATH)/Makefile.kokkos

gpu-stream-kokkos: main.o KOKKOSStream.o
	$(CXX) $(KOKKOS_LDFLAGS) $^ $(KOKKOS_LIBS) -o $@ -DKOKKOS -O3

%.o:%.cpp $(KOKKOS_CPP_DEPENDS)
	$(NVCC_WRAPPER) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) -c $< -DKOKKOS -O3

