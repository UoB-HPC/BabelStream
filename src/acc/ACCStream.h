
// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

#pragma once

#include <iostream>
#include <stdexcept>

#include "Stream.h"

#include <openacc.h>

#define IMPLEMENTATION_STRING "OpenACC"

template <class T>
class ACCStream : public Stream<T>
{

	struct A{
		T *a;
		T *b;
		T *c;
	};

  protected:
    // Size of arrays
    int array_size;
    A aa;
    // Device side pointers
    T *a;
    T *b;
    T *c;

  public:
    ACCStream(const int, int);
    ~ACCStream();

    virtual void copy() override;
    virtual void add() override;
    virtual void mul() override;
    virtual void triad() override;
    virtual void nstream() override;
    virtual T dot() override;

    virtual void init_arrays(T initA, T initB, T initC) override;
    virtual void read_arrays(std::vector<T>& a, std::vector<T>& b, std::vector<T>& c) override;



};
