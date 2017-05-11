CXXFLAGS=-O3

stl-stream: main.cpp STLStream.cpp
	$(CXX) -std=c++17 $(CXXFLAGS) -DSTL $^ $(EXTRA_FLAGS) -o $@

.PHONY: clean
clean:
	rm -f stl-stream

