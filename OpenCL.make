
ocl-stream: main.cpp OCLStream.cpp
	$(CXX) -O3 -std=c++11 -DOCL $^ $(EXTRA_FLAGS) -lOpenCL -o $@

