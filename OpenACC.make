
ifndef COMPILER
define compiler_help
Set COMPILER to ensure correct flags are set.
Available compilers are:
  PGI CRAY
endef
$(info $(compiler_help))
endif

COMPILER_ = $(CXX)
COMPILER_PGI = pgc++
COMPILER_CRAY = CC

FLAGS_ = -O3 -std=c++11

FLAGS_PGI = -std=c++11 -O3 -acc
ifeq ($(COMPILER), PGI)
define target_help
Set a TARGET to ensure PGI targets the correct offload device.
Available targets are:
  SNB, IVB, HSW
  KEPLER, MAXWELL, PASCAL
  HAWAII
endef
ifndef TARGET
$(error $(target_help))
endif
TARGET_FLAGS_SNB     = -ta=multicore -tp=sandybridge
TARGET_FLAGS_IVB     = -ta=multicore -tp=ivybridge
TARGET_FLAGS_HSW     = -ta=multicore -tp=haswell
TARGET_FLAGS_KEPLER  = -ta=nvidia:cc35
TARGET_FLAGS_MAXWELL = -ta=nvidia:cc50
TARGET_FLAGS_PASCAL  = -ta=nvidia:cc60
TARGET_FLAGS_HAWAII  = -ta=radeon:hawaii
ifeq ($(TARGET_FLAGS_$(TARGET)),)
$(error $(target_help))
endif

FLAGS_PGI += $(TARGET_FLAGS_$(TARGET))

endif

FLAGS_CRAY = -hstd=c++11
CXXFLAGS = $(FLAGS_$(COMPILER))

acc-stream: main.cpp ACCStream.cpp
	$(COMPILER_$(COMPILER)) $(CXXFLAGS) -DACC $^ $(EXTRA_FLAGS) -o $@

.PHONY: clean
clean:
	rm -f acc-stream main.o ACCStream.o
