
// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith, Tom Lin
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

#pragma once

#include <iostream>
#include <stdexcept>

#include "Stream.h"


#define IMPLEMENTATION_STRING "Serial"

template <class T>
class SerialStream : public Stream<T>
{
  protected:
    // Size of arrays
    intptr_t array_size;

    // Device side pointers
    T *a;
    T *b;
    T *c;

  public:
    SerialStream(const intptr_t, int);
    ~SerialStream();

    virtual void copy() override;
    virtual void add() override;
    virtual void mul() override;
    virtual void triad() override;
    virtual void nstream() override;
    virtual T dot() override;

    virtual void init_arrays(T initA, T initB, T initC) override;
    virtual void read_arrays(std::vector<T>& a, std::vector<T>& b, std::vector<T>& c) override;
};
