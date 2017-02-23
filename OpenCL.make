
ocl-stream: main.cpp OCLStream.cpp
	$(CXX) -O3 -std=c++11 -DOCL $^ -lOpenCL -o $@

