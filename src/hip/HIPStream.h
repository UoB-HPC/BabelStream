
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

#define IMPLEMENTATION_STRING "HIP"

template <class T>
class HIPStream : public Stream<T>
{
#ifdef __HIP_PLATFORM_NVCC__
  #ifndef DWORDS_PER_LANE
  #define DWORDS_PER_LANE 1
  #endif
#else
  #ifndef DWORDS_PER_LANE
  #define DWORDS_PER_LANE 4
  #endif
#endif
  // Make sure that either:
  //    DWORDS_PER_LANE is less than sizeof(T), in which case we default to 1 element
  //    or
  //    DWORDS_PER_LANE is divisible by sizeof(T)
  static_assert((DWORDS_PER_LANE * sizeof(unsigned int) < sizeof(T)) ||
                (DWORDS_PER_LANE * sizeof(unsigned int) % sizeof(T) == 0),
                "DWORDS_PER_LANE not divisible by sizeof(element_type)");

  static constexpr unsigned int dwords_per_lane{DWORDS_PER_LANE};
  // Take into account the datatype size
  // That is, if we specify 4 DWORDS_PER_LANE, this is 2 FP64 elements
  // and 4 FP32 elements
  static constexpr unsigned int elements_per_lane{
    (DWORDS_PER_LANE * sizeof(unsigned int)) < sizeof(T) ? 1 : (
     DWORDS_PER_LANE * sizeof(unsigned int) / sizeof(T))};

  protected:
    // Size of arrays
    int array_size;
    int block_count;

    // Host array for partial sums for dot kernel
    T *sums;

    // Device side pointers to arrays
    T *d_a;
    T *d_b;
    T *d_c;
    T *d_sum;


  public:

    HIPStream(const int, const int);
    ~HIPStream();

    virtual void copy() override;
    virtual void add() override;
    virtual void mul() override;
    virtual void triad() override;
    virtual void nstream() override;
    virtual T dot() override;

    virtual void init_arrays(T initA, T initB, T initC) override;
    virtual void read_arrays(std::vector<T>& a, std::vector<T>& b, std::vector<T>& c) override;

};
