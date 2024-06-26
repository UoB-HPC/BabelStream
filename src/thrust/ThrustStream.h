// Copyright (c) 2020 Tom Deakin, 2025 Bernhard Manfred Gruber
// University of Bristol HPC, NVIDIA
//
// For full license terms please see the LICENSE file distributed with this
// source code

#pragma once

#include <iostream>
#include <vector>
#include <memory>

#include "Stream.h"

#define IMPLEMENTATION_STRING "Thrust"

template <class T>
class ThrustStream : public Stream<T>
{
  protected:
    struct Impl;
    std::unique_ptr<Impl> impl; // avoid thrust vectors leaking into non-CUDA translation units
    intptr_t array_size;

  public:
    ThrustStream(intptr_t array_size, int device);
    ~ThrustStream();

    virtual void copy() override;
    virtual void add() override;
    virtual void mul() override;
    virtual void triad() override;
    virtual void nstream() override;
    virtual T dot() override;

    virtual void init_arrays(T initA, T initB, T initC) override;
    virtual void read_arrays(std::vector<T>& a, std::vector<T>& b, std::vector<T>& c) override;

};

