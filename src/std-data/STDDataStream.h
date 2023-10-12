// Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.
// Updated 2021 by University of Bristol
//
// For full license terms please see the LICENSE file distributed with this
// source code

#pragma once
#include "dpl_shim.h"

#include <iostream>
#include <stdexcept>
#include "Stream.h"

#define IMPLEMENTATION_STRING "STD (data-oriented)"


template <class T>
class STDDataStream : public Stream<T>
{
  protected:
    // Size of arrays
    int array_size;

    // Device side pointers
    T *a, *b, *c;

  public:
    STDDataStream(const int, int) noexcept;
    ~STDDataStream();

    virtual void copy() override;
    virtual void add() override;
    virtual void mul() override;
    virtual void triad() override;
    virtual void nstream() override;
    virtual T dot() override;

    virtual void init_arrays(T initA, T initB, T initC) override;
    virtual void read_arrays(std::vector<T>& a, std::vector<T>& b, std::vector<T>& c) override;
};

