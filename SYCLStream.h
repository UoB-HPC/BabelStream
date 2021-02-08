
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

#define IMPLEMENTATION_STRING "SYCL"

template <class T>
class SYCLStream : public Stream<T>
{
  protected:
    // Size of arrays
    int array_size;

    // SYCL objects
    std::unique_ptr<sycl::queue> queue;
    std::unique_ptr<sycl::buffer<T>> d_a;
    std::unique_ptr<sycl::buffer<T>> d_b;
    std::unique_ptr<sycl::buffer<T>> d_c;
    std::unique_ptr<sycl::buffer<T>> d_sum;

  public:

    SYCLStream(const int, const int);
    ~SYCLStream() = default;

    virtual void copy() override;
    virtual void add() override;
    virtual void mul() override;
    virtual void triad() override;
    virtual T    dot() override;

    virtual void init_arrays(T initA, T initB, T initC) override;
    virtual void read_arrays(std::vector<T>& a, std::vector<T>& b, std::vector<T>& c) override;

};

// Populate the devices list
void getDeviceList(void);
