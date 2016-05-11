
// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

#pragma once

#include <sstream>

#include "Stream.h"

#include "CL/sycl.hpp"

#define IMPLEMENTATION_STRING "SYCL"

template <class T>
class SYCLStream : public Stream<T>
{
  protected:
    // Size of arrays
    unsigned int array_size;

    // SYCL objects
    cl::sycl::queue *queue;
    cl::sycl::buffer<T> *d_a;
    cl::sycl::buffer<T> *d_b;
    cl::sycl::buffer<T> *d_c;

  public:

    SYCLStream(const unsigned int, const int);
    ~SYCLStream();

    virtual void copy() override;
    virtual void add() override;
    virtual void mul() override;
    virtual void triad() override;

    virtual void write_arrays(const std::vector<T>& a, const std::vector<T>& b, const std::vector<T>& c) override;
    virtual void read_arrays(std::vector<T>& a, std::vector<T>& b, std::vector<T>& c) override;

};

// Populate the devices list
void getDeviceList(void);
