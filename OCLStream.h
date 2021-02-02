
// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

#pragma once

#include <iostream>
#include <sstream>
#include <stdexcept>

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120

#include "CL/cl2.hpp"

#include "Stream.h"

#define IMPLEMENTATION_STRING "OpenCL"

template <class T>
class OCLStream : public Stream<T>
{
  protected:
    // Size of arrays
    int array_size;

    // Host array for partial sums for dot kernel
    std::vector<T> sums;

    // OpenCL objects
    cl::Device device;
    cl::Context context;
    cl::CommandQueue queue;

    // Device side pointers to arrays
    cl::Buffer d_a;
    cl::Buffer d_b;
    cl::Buffer d_c;
    cl::Buffer d_sum;

    cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, T, T, T> *init_kernel;
    cl::KernelFunctor<cl::Buffer, cl::Buffer> *copy_kernel;
    cl::KernelFunctor<cl::Buffer, cl::Buffer> * mul_kernel;
    cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer> *add_kernel;
    cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer> *triad_kernel;
    cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer> *nstream_kernel;
    cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::LocalSpaceArg, cl_int> *dot_kernel;

    // NDRange configuration for the dot kernel
    size_t dot_num_groups;
    size_t dot_wgsize;

  public:

    OCLStream(const int, const int);
    ~OCLStream();

    virtual void copy() override;
    virtual void add() override;
    virtual void mul() override;
    virtual void triad() override;
    virtual void nstream() override;
    virtual T dot() override;

    virtual void init_arrays(T initA, T initB, T initC) override;
    virtual void read_arrays(std::vector<T>& a, std::vector<T>& b, std::vector<T>& c) override;

};

// Populate the devices list
void getDeviceList(void);
