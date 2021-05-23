
ifndef COMPILER
define compiler_help
Set COMPILER to ensure correct flags are set.
Available compilers are:
  PGI GNU
endef
$(info $(compiler_help))
endif

COMPILER_ = $(CXX)
COMPILER_PGI = pgc++
COMPILER_GNU = g++

FLAGS_ = -O3 -std=c++11

FLAGS_PGI = -std=c++11 -O3 -acc
ifeq ($(COMPILER), PGI)
define target_help
Set a TARGET to ensure PGI targets the correct offload device.
Available targets are:
  SNB, IVB, HSW, SKL, KNL
  PWR9, AMD
  KEPLER, MAXWELL, PASCAL, VOLTA
  HAWAII
endef
ifndef TARGET
$(error $(target_help))
endif
TARGET_FLAGS_SNB     = -ta=multicore -tp=sandybridge
TARGET_FLAGS_IVB     = -ta=multicore -tp=ivybridge
TARGET_FLAGS_HSW     = -ta=multicore -tp=haswell
TARGET_FLAGS_SKL     = -ta=multicore -tp=skylake
TARGET_FLAGS_KNL     = -ta=multicore -tp=knl
TARGET_FLAGS_PWR9    = -ta=multicore -tp=pwr9
TARGET_FLAGS_AMD     = -ta=multicore -tp=zen
TARGET_FLAGS_KEPLER  = -ta=nvidia:cc35
TARGET_FLAGS_MAXWELL = -ta=nvidia:cc50
TARGET_FLAGS_PASCAL  = -ta=nvidia:cc60
TARGET_FLAGS_VOLTA   = -ta=nvidia:cc70
TARGET_FLAGS_HAWAII  = -ta=radeon:hawaii
ifeq ($(TARGET_FLAGS_$(TARGET)),)
$(error $(target_help))
endif

FLAGS_PGI += $(TARGET_FLAGS_$(TARGET))

endif

FLAGS_GNU = -O3 -std=c++11 -Drestrict=__restrict -fopenacc
CXXFLAGS = $(FLAGS_$(COMPILER))

acc-stream: main.cpp ACCStream.cpp
	$(COMPILER_$(COMPILER)) $(CXXFLAGS) -DACC $^ $(EXTRA_FLAGS) -o $@

.PHONY: clean
clean:
	rm -f acc-stream main.o ACCStream.o
