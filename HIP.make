
# TODO: HIP with HCC

HIPCC = hipcc

hip-stream: main.cpp HIPStream.cpp
	$(HIPCC) $(CXXFLAGS) -std=c++11 -DHIP $^ $(EXTRA_FLAGS) -o $@

.PHONY: clean
clean:
	rm -f hip-stream

