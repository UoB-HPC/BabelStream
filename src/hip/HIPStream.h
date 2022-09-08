
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
#define DOT_READ_DWORDS_PER_LANE 4


template <class T>
class HIPStream : public Stream<T>
{
  // Make sure that either:
  //    DOT_READ_DWORDS_PER_LANE is less than sizeof(T), in which case we default to 1 element
  //    or
  //    DOT_READ_DWORDS_PER_LANE is divisible by sizeof(T)
  static_assert((DOT_READ_DWORDS_PER_LANE * sizeof(unsigned int) < sizeof(T)) ||
                (DOT_READ_DWORDS_PER_LANE * sizeof(unsigned int) % sizeof(T) == 0),
                "DOT_READ_DWORDS_PER_LANE not divisible by sizeof(element_type)");

  // Take into account the datatype size
  // That is, for 4 DOT_READ_DWORDS_PER_LANE, this is 2 FP64 elements
  // and 4 FP32 elements
  static constexpr unsigned int dot_elements_per_lane{
    (DOT_READ_DWORDS_PER_LANE * sizeof(unsigned int)) < sizeof(T) ? 1 : (
     DOT_READ_DWORDS_PER_LANE * sizeof(unsigned int) / sizeof(T))};

  protected:
    // Size of arrays
    int array_size;
    int dot_num_blocks;

    // Host array for partial sums for dot kernel
    T *sums;

    // Device side pointers to arrays
    T *d_a;
    T *d_b;
    T *d_c;


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
