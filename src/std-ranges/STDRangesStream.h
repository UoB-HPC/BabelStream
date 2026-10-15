// Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.
// Updated 2021 by University of Bristol
//
// For full license terms please see the LICENSE file distributed with this
// source code

#pragma once

#include <iostream>
#include <stdexcept>
#include "Stream.h"

#define IMPLEMENTATION_STRING "STD C++ ranges"


template <class T>
class STDRangesStream : public Stream<T>
{
  protected:
    // Size of arrays
    intptr_t array_size;

    // Device side pointers
    T *a, *b, *c;

  public:
    STDRangesStream(BenchId bs, const intptr_t array_size, const int device_id,
		  T initA, T initB, T initC);
    ~STDRangesStream();

    void copy() override;
    void add() override;
    void mul() override;
    void triad() override;
    void nstream() override;
    T dot() override;

    void get_arrays(T const*& a, T const*& b, T const*& c) override;
    void init_arrays(T initA, T initB, T initC) override;
};
