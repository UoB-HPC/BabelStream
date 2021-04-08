
HCC = hcc

CXXFLAGS+=-O3 $(shell hcc-config --cxxflags)
LDFLAGS+=$(shell hcc-config --ldflags)

ifdef TBSIZE
CXXFLAGS+=-DVIRTUALTILESIZE=$(TBSIZE)
endif

ifdef NTILES
CXXFLAGS+=-DNTILES=$(TBSIZE)
endif


hc-stream: ../main.cpp HCStream.cpp
	$(HCC) $(CXXFLAGS) -DHC  $^  $(LDFLAGS) $(EXTRA_FLAGS) -o $@

.PHONY: clean
clean:
	rm -f hc-stream
