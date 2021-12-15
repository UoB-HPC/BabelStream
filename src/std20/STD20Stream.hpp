// Copyright (c) 2020 Tom Deakin
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

#pragma once

#include <iostream>
#include <vector>

#include "Stream.h"

#define IMPLEMENTATION_STRING "C++20"

template <class T>
class STD20Stream : public Stream<T>
{
  protected:
    // Size of arrays
    int array_size;

    // Device side pointers
    std::vector<T> a;
    std::vector<T> b;
    std::vector<T> c;

  public:
    STD20Stream(const int, int);
    ~STD20Stream() = default;

    virtual void copy() override;
    virtual void add() override;
    virtual void mul() override;
    virtual void triad() override;
    virtual void nstream() override;
    virtual T dot() override;

    virtual void init_arrays(T initA, T initB, T initC) override;
    virtual void read_arrays(std::vector<T>& a, std::vector<T>& b, std::vector<T>& c) override;

};

