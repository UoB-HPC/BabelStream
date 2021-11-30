// Copyright (c) 2020 Tom Deakin
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

#pragma once

#include <iostream>
#include <vector>
#include <thrust/device_vector.h>

#include "Stream.h"

#define IMPLEMENTATION_STRING "Thrust"

template <class T>
class ThrustStream : public Stream<T>
{
  protected:
    // Size of arrays
    int array_size;

    thrust::device_vector<T> a;
    thrust::device_vector<T> b;
    thrust::device_vector<T> c;

  public:
    ThrustStream(const int, int);
    ~ThrustStream() = default;

    virtual void copy() override;
    virtual void add() override;
    virtual void mul() override;
    virtual void triad() override;
    virtual void nstream() override;
    virtual T dot() override;

    virtual void init_arrays(T initA, T initB, T initC) override;
    virtual void read_arrays(std::vector<T>& a, std::vector<T>& b, std::vector<T>& c) override;

};

