// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

#pragma once

#include <iostream>
#include <stdexcept>
#include "RAJA/RAJA.hxx"

#include "Stream.h"

#define IMPLEMENTATION_STRING "RAJA"

#ifdef RAJA_TARGET_CPU
typedef RAJA::IndexSet::ExecPolicy<
        RAJA::seq_segit,
        RAJA::omp_parallel_for_exec> policy;
typedef RAJA::omp_reduce reduce_policy;
#else
const size_t block_size = 128;
typedef RAJA::IndexSet::ExecPolicy<
        RAJA::seq_segit,
        RAJA::cuda_exec<block_size>> policy;
typedef RAJA::cuda_reduce<block_size> reduce_policy;
#endif

template <class T>
class RAJAStream : public Stream<T>
{
  protected:
    // Size of arrays
    unsigned int array_size;

    // Contains iteration space
    RAJA::IndexSet index_set;

    // Device side pointers to arrays
    T* d_a;
    T* d_b;
    T* d_c;

  public:

    RAJAStream(const unsigned int, const int);
    ~RAJAStream();

    virtual void copy() override;
    virtual void add() override;
    virtual void mul() override;
    virtual void triad() override;
    virtual T dot() override;

    virtual void init_arrays(T initA, T initB, T initC) override;
    virtual void read_arrays(
            std::vector<T>& a, std::vector<T>& b, std::vector<T>& c) override;
};

