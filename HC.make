
HCC = hcc

CXXFLAGS+=-O3 -hc -I/opt/rocm/hcc/include #-stdlib=libc++
LDFLAGS+=-L/opt/rocm/hcc/lib -Wl,--whole-archive -lmcwamp -Wl,--no-whole-archive -ldl #-Wl,--rpath=/opt/rocm/hcc/lib -lc++ -lc++abi -ldl

hc-stream: main.cpp HCStream.cpp
	$(HCC) $(CXXFLAGS) -DHC $^  $(LDFLAGS) $(EXTRA_FLAGS) -o $@

.PHONY: clean
clean:
	rm -f hc-stream
