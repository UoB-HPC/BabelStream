
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

namespace sycl_kernels
{
  template <class T> class init;
  template <class T> class copy;
  template <class T> class mul;
  template <class T> class add;
  template <class T> class triad;
  template <class T> class nstream;
  template <class T> class dot;
}

template <class T>
class SYCLStream : public Stream<T>
{
  protected:
    // Size of arrays
    size_t array_size;

    // SYCL objects
    cl::sycl::queue *queue;
    cl::sycl::buffer<T> *d_a;
    cl::sycl::buffer<T> *d_b;
    cl::sycl::buffer<T> *d_c;
    cl::sycl::buffer<T> *d_sum;

    // SYCL kernel names
    typedef sycl_kernels::init<T> init_kernel;
    typedef sycl_kernels::copy<T> copy_kernel;
    typedef sycl_kernels::mul<T> mul_kernel;
    typedef sycl_kernels::add<T> add_kernel;
    typedef sycl_kernels::triad<T> triad_kernel;
    typedef sycl_kernels::nstream<T> nstream_kernel;
    typedef sycl_kernels::dot<T> dot_kernel;

    // NDRange configuration for the dot kernel
    size_t dot_num_groups;
    size_t dot_wgsize;

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
    void init_arrays(T initA, T initB, T initC);
};

// Populate the devices list
void getDeviceList(void);
