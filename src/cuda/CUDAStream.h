
// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

#pragma once

#include <iostream>
#include <stdexcept>
#include <sstream>

#include "Stream.h"

#define IMPLEMENTATION_STRING "CUDA"

#define TBSIZE 256
#define TBSIZE_DOT 1024

template <class T>
class CUDAStream : public Stream<T>
{
  protected:
    // Size of arrays
    intptr_t array_size;

    // Host array for partial sums for dot kernel
    T* sums;

    // Device side pointers to arrays
    T* d_a;
    T* d_b;
    T* d_c;

    // If UVM is disabled, host arrays for verification purposes
    std::vector<T> h_a, h_b, h_c;

    // Number of blocks for dot kernel
    intptr_t dot_num_blocks;

  public:
    CUDAStream(BenchId bs, const intptr_t array_size, const int device_id,
	       T initA, T initB, T initC);
    ~CUDAStream();

    void copy() override;
    void add() override;
    void mul() override;
    void triad() override;
    void nstream() override;
    T dot() override;

    void get_arrays(T const*& a, T const*& b, T const*& c) override;
    void init_arrays(T initA, T initB, T initC);
};
