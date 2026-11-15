
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
#if defined(SVM_COARSE) || defined(SVM_FINE) || defined(SVM_SYSTEM)
#define OCL_SVM
#define CL_HPP_TARGET_OPENCL_VERSION 200
#define CL_HPP_MINIMUM_OPENCL_VERSION 200
#else
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#endif

#include "CL/cl2.hpp"

#include "Stream.h"

#define IMPLEMENTATION_STRING "OpenCL"

template <class T>
class OCLStream : public Stream<T>
{
  protected:
#if defined(OCL_SVM)
    using Array = T*;
#else
    using Array = cl::Buffer;
#endif

    // Size of arrays
    intptr_t array_size;

    // Host array for partial sums for dot kernel
    std::vector<T> sums;

    // OpenCL objects
    cl::Device device;
    cl::Context context;
    cl::CommandQueue queue;

    // Device side pointers to arrays
    Array d_a;
    Array d_b;
    Array d_c;
    cl::Buffer d_sum;

    // Host-side arrays for verification
    std::vector<T> h_a, h_b, h_c;

    cl::KernelFunctor<Array, Array, Array, T, T, T>* init_kernel;
    cl::KernelFunctor<Array, Array>* copy_kernel;
    cl::KernelFunctor<Array, Array>* mul_kernel;
    cl::KernelFunctor<Array, Array, Array>* add_kernel;
    cl::KernelFunctor<Array, Array, Array>* triad_kernel;
    cl::KernelFunctor<Array, Array, Array>* nstream_kernel;
    cl::KernelFunctor<Array, Array, cl::Buffer, cl::LocalSpaceArg, cl_long>* dot_kernel;

    // NDRange configuration for the dot kernel
    size_t dot_num_groups;
    size_t dot_wgsize;

  public:
    OCLStream(BenchId bs, const intptr_t array_size, const int device_id,
        T initA, T initB, T initC);
    ~OCLStream();

    void copy() override;
    void add() override;
    void mul() override;
    void triad() override;
    void nstream() override;
    T dot() override;

    void get_arrays(T const*& a, T const*& b, T const*& c) override;
    void init_arrays(T initA, T initB, T initC) override;
};

// Populate the devices list
void getDeviceList(void);
