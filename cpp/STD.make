# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# For full license terms please see the LICENSE file distributed with this
# source code

CXXFLAGS=-O3 -std=c++17 -stdpar -DSTD
STD_CXX=nvc++

std-stream: main.cpp STDStream.cpp
	$(STD_CXX) $(CXXFLAGS) $^ $(EXTRA_FLAGS) -o $@

.PHONY: clean
clean:
	rm -f std-stream
