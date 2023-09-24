// Copyright (c) 2020 Tom Deakin
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

#pragma once
#include "dpl_shim.h"

#include <iostream>
#include <stdexcept>
#include "Stream.h"

#define IMPLEMENTATION_STRING "STD C++ ranges"

template <class T>
class STDRangesStream : public Stream<T>
{
  protected:
    // Size of arrays
    int array_size;

    // Device side pointers
    T *a, *b, *c;

  public:
    STDRangesStream(const int, int) noexcept;
    ~STDRangesStream();

    virtual void copy() override;
    virtual void add() override;
    virtual void mul() override;
    virtual void triad() override;
    virtual void nstream() override;
    virtual T dot() override;

    virtual void init_arrays(T initA, T initB, T initC) override;
    virtual void read_arrays(std::vector<T>& a, std::vector<T>& b, std::vector<T>& c) override;

};

