
// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

#pragma once

#include <iostream>
#include <stdexcept>

#include "Stream.h"

#include <omp.h>

#define IMPLEMENTATION_STRING "OpenMP 4.5"

template <class T>
class OMP45Stream : public Stream<T>
{
  protected:
    // Size of arrays
    unsigned int array_size;

    // Device side pointers
    T *a;
    T *b;
    T *c;

  public:
    OMP45Stream(const unsigned int, T*, T*, T*, int);
    ~OMP45Stream();

    virtual void copy() override;
    virtual void add() override;
    virtual void mul() override;
    virtual void triad() override;

    virtual void write_arrays(const std::vector<T>& a, const std::vector<T>& b, const std::vector<T>& c) override;
    virtual void read_arrays(std::vector<T>& a, std::vector<T>& b, std::vector<T>& c) override;



};
