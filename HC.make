
# TODO: HC with HCC

HCC = hcc



CXXFLAGS+=-hc -stdlib=libc++ -I/opt/rocm/hcc-lc/include
LDFLAGS+=-hc -L/opt/rocm/hcc-lc/lib -Wl,--rpath=/opt/rocm/hcc-lc/lib -lc++ -lc++abi -ldl -Wl,--whole-archive -lmcwamp -Wl,--no-whole-archive

hc-stream: main.cpp HCStream.cpp
	$(HCC) $(CXXFLAGS) -DHC $^  $(LDFLAGS) $(EXTRA_FLAGS) -o $@

.PHONY: clean
clean:
	rm -f hc-stream
