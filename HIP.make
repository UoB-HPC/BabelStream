
HIP_PATH?= /opt/rocm/hip
HIPCC=$(HIP_PATH)/bin/hipcc

hip-stream: main.cpp HIPStream.cpp
	$(HIPCC) $(CXXFLAGS) -O3 -std=c++11 -DHIP $^ $(EXTRA_FLAGS) -o $@

.PHONY: clean
clean:
	rm -f hip-stream

