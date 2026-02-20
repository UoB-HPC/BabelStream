// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

#pragma once

#include <iostream>
#include <stdexcept>
#include "RAJA/RAJA.hpp"

#include "Stream.h"

#define IMPLEMENTATION_STRING "RAJA"

#ifdef RAJA_TARGET_CPU
// TODO verify old and new templates are semantically equal
//typedef RAJA::ExecPolicy<
//        RAJA::seq_segit,
//        RAJA::omp_parallel_for_exec> policy;

typedef RAJA::omp_parallel_for_exec policy;
typedef RAJA::omp_reduce reduce_policy;
#else
const size_t block_size = 128;
// TODO verify old and new templates are semantically equal
//typedef RAJA::IndexSet::ExecPolicy<
//        RAJA::seq_segit,
//        RAJA::cuda_exec<block_size>> policy;
//typedef RAJA::cuda_reduce<block_size> reduce_policy;
typedef RAJA::cuda_exec<block_size> policy;
typedef RAJA::cuda_reduce reduce_policy;
#endif

using RAJA::RangeSegment;


template <class T>
class RAJAStream : public Stream<T>
{
  protected:
    // Size of arrays
    const intptr_t array_size;
    const RangeSegment range;

    // Device side pointers to arrays
    T* d_a;
    T* d_b;
    T* d_c;

  public:
    RAJAStream(BenchId bs, const intptr_t array_size, const int device_id,
	       T initA, T initB, T initC);
    ~RAJAStream();

    void copy() override;
    void add() override;
    void mul() override;
    void triad() override;
    void nstream() override;
    T dot() override;

    void get_arrays(T const*& a, T const*& b, T const*& c) override;  
    void init_arrays(T initA, T initB, T initC);
};

