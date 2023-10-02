
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

#define IMPLEMENTATION_STRING "SYCL2020 USM"

template <class T>
class SYCLStream : public Stream<T>
{
  protected:
    // Size of arrays
    size_t array_size;

    // SYCL objects
    // Queue is a pointer because we allow device selection
    std::unique_ptr<sycl::queue> queue;

    // Buffers
    T *a{};
    T *b{};
    T *c{};
    T *sum{};

  public:

    SYCLStream(const size_t, const int);
    ~SYCLStream();

    virtual void copy() override;
    virtual void add() override;
    virtual void mul() override;
    virtual void triad() override;
    virtual void nstream() override;
    virtual T    dot() override;

    virtual void init_arrays(T initA, T initB, T initC) override;
    virtual void read_arrays(std::vector<T>& a, std::vector<T>& b, std::vector<T>& c) override;

};

// Populate the devices list
void getDeviceList(void);
