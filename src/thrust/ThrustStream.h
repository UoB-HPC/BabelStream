// Copyright (c) 2020 Tom Deakin, 2025 Bernhard Manfred Gruber
// University of Bristol HPC, NVIDIA
//
// For full license terms please see the LICENSE file distributed with this
// source code

#pragma once

#include <iostream>
#include <vector>
#include <memory>
#include <cstdint>

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

    void copy() override;
    void add() override;
    void mul() override;
    void triad() override;
    void nstream() override;
    T dot() override;

    void get_arrays(T const*& a, T const*& b, T const*& c) override;
    void init_arrays(T initA, T initB, T initC);
};

