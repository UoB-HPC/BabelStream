// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
//
// For full license terms please see the LICENSE file distributed with this
// source code

#pragma once

#include <iostream>
#include <stdexcept>
#include "Stream.h"

#define IMPLEMENTATION_STRING "STD"

template <class T>
class STDStream : public Stream<T>
{
  protected:
    // Size of arrays
    int array_size;

    // Device side pointers
    T *a;
    T *b;
    T *c;

  public:
    STDStream(const int, int) noexcept;
    ~STDStream();

    virtual void copy() override;
    virtual void add() override;
    virtual void mul() override;
    virtual void triad() override;
    virtual void nstream() override;
    virtual T dot() override;

    virtual void init_arrays(T initA, T initB, T initC) override;
    virtual void read_arrays(std::vector<T>& a, std::vector<T>& b, std::vector<T>& c) override;
};

