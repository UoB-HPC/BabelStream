
// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

#pragma once

#include <sstream>
#include <memory>

#include "Stream.h"

#include <sycl/sycl.hpp>

#define IMPLEMENTATION_STRING "SYCL2020 "

template <class T>
class SYCLStream : public Stream<T>
{
  protected:
    // Size of arrays
    size_t array_size;
    bool use_shared_alloc;

    // SYCL objects
    // Queue is a pointer because we allow device selection
    std::unique_ptr<sycl::queue> queue;

    // Buffers
    T *a, *b, *c, *sum{};

    // used to implement get_arrays for device USM
    std::vector<T> hostA, hostB, hostC;

  public:

    SYCLStream(BenchId bs, const intptr_t array_size, const int device_id,
	       T initA, T initB, T initC);
    ~SYCLStream();

    void copy() override;
    void add() override;
    void mul() override;
    void triad() override;
    void nstream() override;
    T    dot() override;

    void get_arrays(T const*& a, T const*& b, T const*& c) override;
    void init_arrays(T initA, T initB, T initC) override;
};

// Populate the devices list
void getDeviceList(void);
