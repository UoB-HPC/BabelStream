
default: gpu-stream-kokkos

include $(KOKKOS_PATH)/Makefile.kokkos

gpu-stream-kokkos: main.o KOKKOSStream.o
	$(CXX) $(KOKKOS_LDFLAGS) $^ $(KOKKOS_LIBS) -o $@ -DKOKKOS -DKOKKOS_TARGET_CPU -O3

%.o:%.cpp $(KOKKOS_CPP_DEPENDS)
	$(CXX) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) -c $< -DKOKKOS -DKOKKOS_TARGET_CPU -O3
